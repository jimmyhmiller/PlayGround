//! Spatial queries over the scene.
//!
//! Reimplemented in Rust from Excalidraw's selection/hit logic. Upstream
//! derivation:
//! - `packages/excalidraw/scene/comparisons.ts` and
//!   `packages/excalidraw/element/collision.ts` (Excalidraw, MIT) — the
//!   `getElementsAtPosition` / box-selection scans.
//!
//! These are deliberately simple O(n) linear scans over each element's raw
//! (axis-aligned, unrotated) box. They are the coarse phase; precise, rotation-
//! and curve-aware hit-testing lives in `geometry::hit` and is *not* duplicated
//! here. `topmost_at` reports the frontmost element whose raw box contains the
//! point — callers needing pixel-tight hits refine the result through
//! `geometry::hit`.

use crate::element::ElementId;
use crate::geometry::{Point, Rect};
use crate::scene::Scene;

impl Scene {
    /// Live elements whose raw box intersects `rect`, in paint order (bottom
    /// first). Touching edges count as intersecting (matches `Rect::intersects`).
    pub fn elements_in_rect(&self, rect: &Rect) -> Vec<ElementId> {
        self.iter_live()
            .filter(|e| rect.intersects(&e.raw_box()))
            .map(|e| e.id.clone())
            .collect()
    }

    /// Live elements whose raw box is *fully contained* by `rect`, in paint
    /// order. This is the strict box-selection variant (Excalidraw's "drag a
    /// selection box and only grab fully-enclosed elements").
    pub fn elements_contained_by_rect(&self, rect: &Rect) -> Vec<ElementId> {
        self.iter_live()
            .filter(|e| rect.contains_rect(&e.raw_box()))
            .map(|e| e.id.clone())
            .collect()
    }

    /// The topmost (frontmost in paint order) live element whose raw box contains
    /// `point`, if any. This is the coarse hit; refine with `geometry::hit` for a
    /// pixel-tight result.
    pub fn topmost_at(&self, point: Point) -> Option<ElementId> {
        // Paint order is bottom-first, so the last match is the topmost.
        // `iter_live` is filtered through a HashMap lookup and isn't reliably a
        // DoubleEndedIterator, so fold to keep the last match in a single pass.
        self.iter_live()
            .filter(|e| e.raw_box().contains(point))
            .fold(None, |_, e| Some(e.id.clone()))
    }

    /// All live elements whose raw box contains `point`, ordered top-first
    /// (frontmost first). Useful for alt-click cycling through stacked elements.
    pub fn elements_at(&self, point: Point) -> Vec<ElementId> {
        let mut hits: Vec<ElementId> = self
            .iter_live()
            .filter(|e| e.raw_box().contains(point))
            .map(|e| e.id.clone())
            .collect();
        hits.reverse(); // top-first
        hits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementKind};

    fn rect(id: &str, x: f64, y: f64, w: f64, h: f64) -> Element {
        Element::new(ElementId::from(id), 1, x, y, w, h, ElementKind::Rectangle)
    }

    fn scene(elements: Vec<Element>) -> Scene {
        let mut s = Scene::new();
        for e in elements {
            s.insert(e);
        }
        s
    }

    fn ids(v: Vec<ElementId>) -> Vec<String> {
        v.iter().map(|i| i.as_str().to_string()).collect()
    }

    #[test]
    fn in_rect_returns_intersecting_in_paint_order() {
        let s = scene(vec![
            rect("a", 0.0, 0.0, 10.0, 10.0),
            rect("b", 100.0, 100.0, 10.0, 10.0),
            rect("c", 5.0, 5.0, 10.0, 10.0),
        ]);
        // Query box around the origin: a and c intersect, b does not.
        let q = Rect::new(0.0, 0.0, 12.0, 12.0);
        assert_eq!(ids(s.elements_in_rect(&q)), ["a", "c"]);
    }

    #[test]
    fn in_rect_touching_edge_counts() {
        let s = scene(vec![rect("a", 10.0, 10.0, 10.0, 10.0)]);
        // Query box whose right edge exactly touches a's left edge.
        let q = Rect::new(0.0, 10.0, 10.0, 10.0);
        assert_eq!(ids(s.elements_in_rect(&q)), ["a"]);
    }

    #[test]
    fn contained_by_rect_is_strict() {
        let s = scene(vec![
            rect("inside", 10.0, 10.0, 5.0, 5.0),
            rect("straddle", 18.0, 18.0, 10.0, 10.0),
        ]);
        let q = Rect::new(0.0, 0.0, 20.0, 20.0);
        // Only fully-enclosed element returned.
        assert_eq!(ids(s.elements_contained_by_rect(&q)), ["inside"]);
    }

    #[test]
    fn topmost_at_returns_frontmost() {
        // a and b overlap at (5,5); b is painted later, so it's on top.
        let s = scene(vec![
            rect("a", 0.0, 0.0, 10.0, 10.0),
            rect("b", 0.0, 0.0, 10.0, 10.0),
        ]);
        assert_eq!(
            s.topmost_at(Point::new(5.0, 5.0)),
            Some(ElementId::from("b"))
        );
    }

    #[test]
    fn topmost_respects_reordering() {
        let mut s = scene(vec![
            rect("a", 0.0, 0.0, 10.0, 10.0),
            rect("b", 0.0, 0.0, 10.0, 10.0),
        ]);
        // Give indices and bring a to front.
        s.assign_initial_indices().unwrap();
        s.bring_to_front(&[ElementId::from("a")]).unwrap();
        assert_eq!(
            s.topmost_at(Point::new(5.0, 5.0)),
            Some(ElementId::from("a"))
        );
    }

    #[test]
    fn topmost_none_when_empty_or_miss() {
        let s = scene(vec![rect("a", 0.0, 0.0, 10.0, 10.0)]);
        assert_eq!(s.topmost_at(Point::new(50.0, 50.0)), None);
        let empty = Scene::new();
        assert_eq!(empty.topmost_at(Point::ORIGIN), None);
    }

    #[test]
    fn elements_at_is_top_first() {
        let s = scene(vec![
            rect("a", 0.0, 0.0, 10.0, 10.0),
            rect("b", 0.0, 0.0, 10.0, 10.0),
            rect("c", 0.0, 0.0, 10.0, 10.0),
        ]);
        assert_eq!(ids(s.elements_at(Point::new(5.0, 5.0))), ["c", "b", "a"]);
    }

    #[test]
    fn queries_skip_deleted() {
        let mut a = rect("a", 0.0, 0.0, 10.0, 10.0);
        a.is_deleted = true;
        let s = scene(vec![a, rect("b", 0.0, 0.0, 10.0, 10.0)]);
        assert_eq!(
            ids(s.elements_in_rect(&Rect::new(0.0, 0.0, 10.0, 10.0))),
            ["b"]
        );
        assert_eq!(
            s.topmost_at(Point::new(5.0, 5.0)),
            Some(ElementId::from("b"))
        );
    }
}
