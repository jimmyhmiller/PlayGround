//! Frame-clip planning for the tessellator.
//!
//! Reimplemented in Rust from Excalidraw's frame rendering. Upstream derivation:
//! - `packages/excalidraw/renderer/staticScene.ts` (Excalidraw, MIT) — the static
//!   renderer wraps a frame's children in `context.save()/clip(frameClipRect)/
//!   context.restore()` so children never paint outside the frame box.
//! - `packages/excalidraw/frame.ts` (Excalidraw, MIT) — `getFrameChildren`,
//!   frame clip region (the frame's raw box).
//!
//! Excalidraw frames are clipping containers: every element that belongs to a
//! frame (its `frame_id` points at the frame) is clipped to the frame's box.
//! The frame's own outline still draws unclipped; only its children are masked.
//!
//! # Ordering assumption
//!
//! The scene stores a single flat, paint-ordered element list. We rely on
//! Excalidraw's invariant that **a frame's children are contiguous in paint
//! order** — they form an unbroken run, painted directly above their frame
//! element. Excalidraw maintains this via `reorderElements` /
//! `getElementsInResizingFrame` so a frame and its children stay adjacent in the
//! z-order; mixing another frame's children (or unframed elements) into the
//! middle of a run is not a representable state. Under that invariant, clipping
//! reduces to: walk the elements once, and whenever the *effective* frame id of
//! the current element changes, close the previous clip and open the next one.
//!
//! The plan is expressed as per-position [`ClipEdge`]s the tessellator applies
//! around each element it emits, so push/pop are always balanced even when a
//! frame has zero children (no clip is opened in that case).

use crate::element::{Element, ElementId};
use crate::geometry::Rect;
use crate::scene::Scene;

/// What clip transitions happen at one element position in paint order.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct ClipEdge {
    /// If set, emit `PushClip(rect)` *before* this element is drawn.
    pub open: Option<Rect>,
    /// If `true`, emit `PopClip` *after* this element (and any later siblings up
    /// to here) have been drawn — i.e. the run of framed children ends here.
    pub close_before_open: bool,
}

/// Compute the clip edges for a paint-ordered slice of live elements.
///
/// `elements[i]` corresponds to `edges[i]`. For each element we decide whether
/// the run of framed children it sits in begins or ends here:
///
/// - `open` is set on the *first* child of a frame's contiguous run; the rect is
///   the frame's clip rect.
/// - `close_before_open` is set on the element whose effective frame differs from
///   the previous element's, meaning the previous run (if any) must be popped
///   before this element draws. The caller also pops any still-open clip after
///   the final element.
///
/// A frame element itself is treated as having no effective frame (its outline is
/// drawn unclipped). An element whose `frame_id` does not resolve to a live frame
/// in `scene` is also treated as unframed, so a dangling link never opens a clip.
pub(crate) fn plan_clips(scene: &Scene, elements: &[&Element]) -> Vec<ClipEdge> {
    let mut edges = vec![ClipEdge::default(); elements.len()];
    let mut current: Option<ElementId> = None;

    for (i, el) in elements.iter().enumerate() {
        let effective = effective_frame(scene, el);

        if effective.as_ref() != current.as_ref() {
            // The run changes here: close the previous run (if one is open) and
            // open the new one (if this element is framed).
            if current.is_some() {
                edges[i].close_before_open = true;
            }
            if let Some(frame_id) = &effective {
                // Only open a clip we can actually compute a rect for; a live
                // frame always yields one (checked by `effective_frame`).
                if let Some(rect) = scene.frame_clip_rect(frame_id) {
                    edges[i].open = Some(rect);
                }
            }
            current = effective;
        }
    }

    edges
}

/// Whether a still-open clip remains after the final element, i.e. the last
/// element in paint order was a framed child. The tessellator pops once more in
/// that case to keep push/pop balanced.
pub(crate) fn trailing_clip_open(scene: &Scene, elements: &[&Element]) -> bool {
    elements
        .last()
        .and_then(|el| effective_frame(scene, el))
        .is_some()
}

/// The frame this element is clipped to, or `None` if it is unframed, is itself a
/// frame, or its `frame_id` does not point at a live frame.
fn effective_frame(scene: &Scene, el: &Element) -> Option<ElementId> {
    // A frame draws its own outline unclipped; never clip a frame to itself.
    if matches!(el.kind, crate::element::ElementKind::Frame(_)) {
        return None;
    }
    let frame_id = el.frame_id.as_ref()?;
    if scene.is_frame(frame_id) {
        Some(frame_id.clone())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementId, ElementKind};

    fn rect(id: &str, frame: Option<&str>) -> Element {
        let mut e = Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Rectangle,
        );
        e.frame_id = frame.map(ElementId::from);
        e
    }

    fn frame_kind() -> ElementKind {
        serde_json::from_value(serde_json::json!({ "type": "frame", "name": null }))
            .expect("frame kind deserializes")
    }

    fn frame(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, frame_kind())
    }

    fn scene_of(els: Vec<Element>) -> Scene {
        let mut s = Scene::new();
        for e in els {
            s.insert(e);
        }
        s
    }

    #[test]
    fn unframed_elements_have_no_edges() {
        let s = scene_of(vec![rect("a", None), rect("b", None)]);
        let live: Vec<&Element> = s.iter_live().collect();
        let edges = plan_clips(&s, &live);
        assert_eq!(edges, vec![ClipEdge::default(); 2]);
        assert!(!trailing_clip_open(&s, &live));
    }

    #[test]
    fn frame_run_opens_once_and_closes_after() {
        // Order: frame, child1, child2, then an unframed trailing element.
        let s = scene_of(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            rect("c1", Some("f")),
            rect("c2", Some("f")),
            rect("after", None),
        ]);
        let live: Vec<&Element> = s.iter_live().collect();
        let edges = plan_clips(&s, &live);

        // frame element: no clip on itself.
        assert_eq!(edges[0], ClipEdge::default());
        // c1: opens the clip with the frame's rect.
        assert_eq!(edges[1].open, Some(Rect::new(0.0, 0.0, 100.0, 100.0)));
        assert!(!edges[1].close_before_open);
        // c2: still inside the same run, no transition.
        assert_eq!(edges[2], ClipEdge::default());
        // after: run ends, close before this element draws.
        assert!(edges[3].close_before_open);
        assert_eq!(edges[3].open, None);
        assert!(!trailing_clip_open(&s, &live));
    }

    #[test]
    fn trailing_frame_run_stays_open() {
        let s = scene_of(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            rect("c1", Some("f")),
        ]);
        let live: Vec<&Element> = s.iter_live().collect();
        let edges = plan_clips(&s, &live);
        assert_eq!(edges[1].open, Some(Rect::new(0.0, 0.0, 100.0, 100.0)));
        // The last element is a framed child, so a clip is still open at the end.
        assert!(trailing_clip_open(&s, &live));
    }

    #[test]
    fn adjacent_frames_switch_clips() {
        let s = scene_of(vec![
            frame("f", 0.0, 0.0, 50.0, 50.0),
            rect("c1", Some("f")),
            frame("g", 100.0, 0.0, 50.0, 50.0),
            rect("c2", Some("g")),
        ]);
        let live: Vec<&Element> = s.iter_live().collect();
        let edges = plan_clips(&s, &live);
        // c1 opens f's clip.
        assert_eq!(edges[1].open, Some(Rect::new(0.0, 0.0, 50.0, 50.0)));
        // g (a frame) closes the previous run; it opens nothing for itself.
        assert!(edges[2].close_before_open);
        assert_eq!(edges[2].open, None);
        // c2 opens g's clip (no extra close — the run already closed at g).
        assert_eq!(edges[3].open, Some(Rect::new(100.0, 0.0, 50.0, 50.0)));
        assert!(!edges[3].close_before_open);
        assert!(trailing_clip_open(&s, &live));
    }

    #[test]
    fn dangling_frame_id_does_not_clip() {
        // `frame_id` points at a non-existent / non-frame element: treat unframed.
        let s = scene_of(vec![rect("a", Some("ghost"))]);
        let live: Vec<&Element> = s.iter_live().collect();
        let edges = plan_clips(&s, &live);
        assert_eq!(edges, vec![ClipEdge::default(); 1]);
        assert!(!trailing_clip_open(&s, &live));
    }

    #[test]
    fn frame_with_no_children_opens_nothing() {
        let s = scene_of(vec![
            frame("f", 0.0, 0.0, 100.0, 100.0),
            rect("after", None),
        ]);
        let live: Vec<&Element> = s.iter_live().collect();
        let edges = plan_clips(&s, &live);
        // No element opens a clip; nothing to close.
        assert!(edges
            .iter()
            .all(|e| e.open.is_none() && !e.close_before_open));
        assert!(!trailing_clip_open(&s, &live));
    }
}
