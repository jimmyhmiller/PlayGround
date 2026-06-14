//! Frame membership, children queries, and clip rects.
//!
//! Reimplemented in Rust from Excalidraw's frame logic. Upstream derivation:
//! - `packages/excalidraw/frame.ts` (Excalidraw, MIT) — `getElementsInFrame`,
//!   `getFrameChildren`, `elementsAreInFrameBounds`, and the frame clip region.
//!
//! A frame is an [`ElementKind::Frame`] element. Two notions of membership exist
//! and both are useful:
//! - *explicit*: an element whose `frame_id` points at the frame (the persisted
//!   parent link Excalidraw stores once an element is dropped into a frame);
//! - *geometric*: an element whose raw box currently falls inside the frame's
//!   bounds (used while dragging, to decide what would become a child).
//!
//! The frame's clip rect is simply its raw box: a renderer clips children to it.

use crate::element::{ElementId, ElementKind};
use crate::geometry::Rect;
use crate::scene::Scene;

impl Scene {
    /// Whether `id` refers to a live frame element.
    pub fn is_frame(&self, id: &ElementId) -> bool {
        self.get(id)
            .is_some_and(|e| matches!(e.kind, ElementKind::Frame(_)) && !e.is_deleted)
    }

    /// The clip rect of a frame: the axis-aligned region a renderer should clip
    /// the frame's children to. Returns `None` if `id` is not a live frame.
    pub fn frame_clip_rect(&self, id: &ElementId) -> Option<Rect> {
        self.get(id).and_then(|e| {
            if matches!(e.kind, ElementKind::Frame(_)) && !e.is_deleted {
                Some(e.raw_box())
            } else {
                None
            }
        })
    }

    /// The *explicit* children of a frame: live, non-frame elements whose
    /// `frame_id` points at this frame, in current paint order.
    pub fn frame_children(&self, frame: &ElementId) -> Vec<ElementId> {
        if !self.is_frame(frame) {
            return Vec::new();
        }
        self.iter_live()
            .filter(|e| !matches!(e.kind, ElementKind::Frame(_)))
            .filter(|e| e.frame_id.as_ref() == Some(frame))
            .map(|e| e.id.clone())
            .collect()
    }

    /// The *geometric* members of a frame: live, non-frame elements whose raw box
    /// is fully contained by the frame's bounds. This is what a drag-into-frame
    /// gesture uses to decide candidate children, independent of any persisted
    /// `frame_id`. Returns ids in current paint order.
    pub fn elements_in_frame_bounds(&self, frame: &ElementId) -> Vec<ElementId> {
        let Some(clip) = self.frame_clip_rect(frame) else {
            return Vec::new();
        };
        self.iter_live()
            .filter(|e| !matches!(e.kind, ElementKind::Frame(_)))
            .filter(|e| e.id != *frame)
            .filter(|e| clip.contains_rect(&e.raw_box()))
            .map(|e| e.id.clone())
            .collect()
    }

    /// Whether `element` falls inside `frame`'s bounds (full containment of the
    /// element's raw box). `false` if `frame` is not a live frame or the element
    /// does not exist.
    pub fn element_in_frame_bounds(&self, element: &ElementId, frame: &ElementId) -> bool {
        let Some(clip) = self.frame_clip_rect(frame) else {
            return false;
        };
        self.get(element)
            .is_some_and(|e| !e.is_deleted && clip.contains_rect(&e.raw_box()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::Element;

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    /// `FrameData` is not re-exported from `element`, so build the frame payload
    /// through its serde representation (`{"type":"frame","name":null}`) — the
    /// same shape `.excalidraw` files use.
    fn frame_kind() -> ElementKind {
        serde_json::from_value(serde_json::json!({ "type": "frame", "name": null }))
            .expect("frame kind deserializes")
    }

    fn frame(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, frame_kind())
    }

    fn scene(elements: Vec<Element>) -> Scene {
        let mut s = Scene::new();
        for e in elements {
            s.insert(e);
        }
        s
    }

    #[test]
    fn is_frame_and_clip_rect() {
        let s = scene(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            rect("r", 5.0, 5.0, 10.0, 10.0),
        ]);
        assert!(s.is_frame(&ElementId::from("f")));
        assert!(!s.is_frame(&ElementId::from("r")));
        assert_eq!(
            s.frame_clip_rect(&ElementId::from("f")),
            Some(Rect::new(0.0, 0.0, 100.0, 100.0))
        );
        assert_eq!(s.frame_clip_rect(&ElementId::from("r")), None);
    }

    #[test]
    fn elements_in_frame_bounds_requires_full_containment() {
        let s = scene(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            rect("inside", 10.0, 10.0, 20.0, 20.0), // fully inside
            rect("straddle", 90.0, 90.0, 20.0, 20.0), // pokes out the corner
            rect("outside", 200.0, 200.0, 5.0, 5.0), // far away
        ]);
        let ids: Vec<String> = s
            .elements_in_frame_bounds(&ElementId::from("f"))
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids, ["inside"]);
        assert!(s.element_in_frame_bounds(&ElementId::from("inside"), &ElementId::from("f")));
        assert!(!s.element_in_frame_bounds(&ElementId::from("straddle"), &ElementId::from("f")));
    }

    #[test]
    fn frame_does_not_contain_itself() {
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0)]);
        assert!(s.elements_in_frame_bounds(&ElementId::from("f")).is_empty());
    }

    #[test]
    fn explicit_children_by_frame_id() {
        let mut child = rect("c", 5.0, 5.0, 10.0, 10.0);
        child.frame_id = Some(ElementId::from("f"));
        let mut other = rect("o", 5.0, 5.0, 10.0, 10.0);
        other.frame_id = Some(ElementId::from("g")); // different frame
        let s = scene(vec![frame("f", 0.0, 0.0, 100.0, 100.0), child, other]);
        let ids: Vec<String> = s
            .frame_children(&ElementId::from("f"))
            .iter()
            .map(|i| i.as_str().to_string())
            .collect();
        assert_eq!(ids, ["c"]);
    }

    #[test]
    fn non_frame_has_no_children_or_clip() {
        let s = scene(vec![rect("r", 0.0, 0.0, 10.0, 10.0)]);
        assert!(s.frame_children(&ElementId::from("r")).is_empty());
        assert!(s.elements_in_frame_bounds(&ElementId::from("r")).is_empty());
    }

    #[test]
    fn deleted_frame_has_no_clip() {
        let mut f = frame("f", 0.0, 0.0, 100.0, 100.0);
        f.is_deleted = true;
        let s = scene(vec![f]);
        assert!(!s.is_frame(&ElementId::from("f")));
        assert_eq!(s.frame_clip_rect(&ElementId::from("f")), None);
    }
}
