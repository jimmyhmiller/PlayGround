//! Frame membership by geometric containment: the pure helpers the interaction
//! layer calls after a move/resize to decide which frame (if any) an element
//! now belongs to.
//!
//! Reimplemented from Excalidraw's `packages/excalidraw/frame.ts` — specifically
//! the containment side of `getFrameChildren` / `elementsAreInFrameBounds` and
//! the membership-reassignment performed by `addElementsToFrame` /
//! `removeElementsFromFrame` after a drag. We do not vendor any JavaScript.
//!
//! Where [`scene::frames`](super::frames) answers *given a frame, which elements
//! are inside it*, this module answers the inverse the mover needs: *given an
//! element, which frame should it belong to now*, returning only the **change**
//! so the caller applies it (and records undo) exactly once.
//!
//! Both functions are **pure**: they borrow the [`Scene`] immutably and never
//! mutate. Containment uses the element's tight [`element_bounds`] (so rotation
//! is accounted for) against the frame's clip rect.

use crate::element::{Element, ElementId, ElementKind};
use crate::geometry::element_bounds;
use crate::scene::Scene;

/// The topmost frame whose clip rect **fully contains** `element`'s tight
/// bounds, or `None` if no frame contains it.
///
/// Port of Excalidraw's "which frame is this element in" containment check
/// (`elementsAreInFrameBounds` applied per element). We iterate live frames in
/// paint order and keep the last (topmost) one whose [`frame_clip_rect`] fully
/// contains the element's [`element_bounds`]. A frame never contains itself, and
/// deleted elements/frames are skipped (handled via [`Scene::is_frame`] /
/// `iter_live`).
///
/// Containment is *full*: an element straddling a frame edge is **not** inside
/// it, matching upstream (and [`Scene::elements_in_frame_bounds`]).
pub fn frame_containing(scene: &Scene, element: &Element) -> Option<ElementId> {
    if element.is_deleted {
        return None;
    }
    let bounds = element_bounds(element);
    scene
        .iter_live()
        .filter(|f| matches!(f.kind, ElementKind::Frame(_)))
        .filter(|f| f.id != element.id)
        .filter(|f| {
            scene
                .frame_clip_rect(&f.id)
                .is_some_and(|clip| clip.contains_rect(&bounds))
        })
        .map(|f| f.id.clone())
        // `iter_live` yields bottom-first; `last` lands on the topmost frame.
        .last()
}

/// Decide whether `id`'s `frame_id` **should change**, returning:
/// - `Some(Some(frame))` — the element should *join* `frame`;
/// - `Some(None)` — the element should *leave* its current frame;
/// - `None` — no change (already correct, or the element is not a membership
///   candidate: it does not exist, is deleted, or is itself a frame).
///
/// This is what the mover calls per moved element after a commit: a `Some(_)`
/// result tells the caller to set `frame_id` and record undo; a `None` result
/// means do nothing (so undo history stays free of no-op entries).
///
/// Port of the reassignment Excalidraw performs in `addElementsToFrame` /
/// `removeElementsFromFrame` after a drag settles: compute the geometrically
/// containing frame and compare against the stored `frame_id`. Frames are never
/// assigned into other frames here (nested-frame parenting is out of scope for
/// this containment helper), matching the `!isFrameElement` guard upstream.
pub fn assign_frame_membership(scene: &Scene, id: &ElementId) -> Option<Option<ElementId>> {
    let element = scene.get(id)?;
    if element.is_deleted {
        return None;
    }
    // Frames are not assigned into frames by this containment helper.
    if matches!(element.kind, ElementKind::Frame(_)) {
        return None;
    }

    let desired = frame_containing(scene, element);
    if desired == element.frame_id {
        None
    } else {
        Some(desired)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    /// `FrameData` is not re-exported from `element`; build the frame payload
    /// through its serde representation (matches `scene::frames` tests).
    fn frame_kind() -> ElementKind {
        serde_json::from_value(serde_json::json!({ "type": "frame", "name": null }))
            .expect("frame kind deserializes")
    }

    fn frame(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, frame_kind())
    }

    fn scene(els: Vec<Element>) -> Scene {
        let mut s = Scene::new();
        for e in els {
            s.insert(e);
        }
        s
    }

    #[test]
    fn fully_inside_element_joins_frame() {
        let s = scene(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            rect("r", 10.0, 10.0, 20.0, 20.0), // box (10,10)-(30,30), fully inside
        ]);
        assert_eq!(
            frame_containing(&s, s.get(&ElementId::from("r")).unwrap()),
            Some(ElementId::from("f"))
        );
        // frame_id is currently None → should change to Some(f).
        assert_eq!(
            assign_frame_membership(&s, &ElementId::from("r")),
            Some(Some(ElementId::from("f")))
        );
    }

    #[test]
    fn straddling_edge_does_not_join() {
        let s = scene(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            // Box (90,90)-(110,110): pokes out the bottom-right corner.
            rect("r", 90.0, 90.0, 20.0, 20.0),
        ]);
        assert_eq!(
            frame_containing(&s, s.get(&ElementId::from("r")).unwrap()),
            None
        );
        // frame_id None, desired None → no change.
        assert_eq!(assign_frame_membership(&s, &ElementId::from("r")), None);
    }

    #[test]
    fn moving_out_clears_membership() {
        // Element currently a member of f, but its box is now far outside.
        let mut r = rect("r", 200.0, 200.0, 10.0, 10.0);
        r.frame_id = Some(ElementId::from("f"));
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0), r]);
        // Desired containing frame is None, but frame_id is Some(f) → leave.
        assert_eq!(
            assign_frame_membership(&s, &ElementId::from("r")),
            Some(None)
        );
    }

    #[test]
    fn already_correct_membership_is_no_change() {
        let mut r = rect("r", 10.0, 10.0, 20.0, 20.0);
        r.frame_id = Some(ElementId::from("f"));
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0), r]);
        // Inside f and already frame_id == f → no change.
        assert_eq!(assign_frame_membership(&s, &ElementId::from("r")), None);
    }

    #[test]
    fn topmost_frame_wins_when_nested_bounds_overlap() {
        // Two overlapping frames both containing the element; the topmost
        // (later-inserted) one is chosen.
        let s = scene(vec![
            frame("outer", 0.0, 0.0, 200.0, 200.0),
            frame("inner", 0.0, 0.0, 100.0, 100.0),
            rect("r", 10.0, 10.0, 20.0, 20.0),
        ]);
        assert_eq!(
            frame_containing(&s, s.get(&ElementId::from("r")).unwrap()),
            Some(ElementId::from("inner"))
        );
    }

    #[test]
    fn frame_is_never_assigned_into_another_frame() {
        // A small frame fully inside a big frame: containment helper still
        // returns None for the inner frame (frames are not members).
        let s = scene(vec![
            frame("big", 0.0, 0.0, 200.0, 200.0),
            frame("small", 10.0, 10.0, 20.0, 20.0),
        ]);
        assert_eq!(assign_frame_membership(&s, &ElementId::from("small")), None);
    }

    #[test]
    fn deleted_element_is_skipped() {
        let mut r = rect("r", 10.0, 10.0, 20.0, 20.0);
        r.is_deleted = true;
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0), r]);
        assert_eq!(
            frame_containing(&s, s.get(&ElementId::from("r")).unwrap()),
            None
        );
        assert_eq!(assign_frame_membership(&s, &ElementId::from("r")), None);
    }

    #[test]
    fn missing_element_yields_none() {
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0)]);
        assert_eq!(assign_frame_membership(&s, &ElementId::from("ghost")), None);
    }

    #[test]
    fn no_frame_in_scene_yields_no_change() {
        let s = scene(vec![rect("r", 10.0, 10.0, 20.0, 20.0)]);
        assert_eq!(
            frame_containing(&s, s.get(&ElementId::from("r")).unwrap()),
            None
        );
        assert_eq!(assign_frame_membership(&s, &ElementId::from("r")), None);
    }

    #[test]
    fn rotated_element_uses_tight_bounds() {
        // A 20x20 rect at (40,40) rotated 45°: its tight bounds grow to roughly
        // 28.28 across, centered at (50,50) → about (35.86,35.86)-(64.14,64.14),
        // still inside a (0,0)-(100,100) frame.
        let mut r = rect("r", 40.0, 40.0, 20.0, 20.0);
        r.angle = std::f64::consts::FRAC_PI_4;
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0), r]);
        assert_eq!(
            frame_containing(&s, s.get(&ElementId::from("r")).unwrap()),
            Some(ElementId::from("f"))
        );

        // The same rotated rect near the frame edge now straddles it once its
        // tight (expanded) bounds cross the boundary.
        let mut r2 = rect("r2", 88.0, 50.0, 20.0, 20.0);
        r2.angle = std::f64::consts::FRAC_PI_4;
        let s2 = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0), r2]);
        assert_eq!(
            frame_containing(&s2, s2.get(&ElementId::from("r2")).unwrap()),
            None
        );
    }
}
