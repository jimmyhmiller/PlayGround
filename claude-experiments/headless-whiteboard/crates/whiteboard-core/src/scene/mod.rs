//! The scene: the authoritative store of elements and their ordering.
//!
//! Phase 1 expands this with groups, frames, fractional-index z-ordering and
//! spatial queries. This file establishes the store API the rest of the library
//! codes against: insertion order *is* paint order until fractional indexing
//! lands, and elements are addressed by [`ElementId`].

use crate::element::{Element, ElementId};
use crate::geometry::Rect;
use std::collections::HashMap;

/// The element store. Owns every element and defines paint order.
#[derive(Debug, Clone, Default)]
pub struct Scene {
    /// Paint order, bottom first. Holds the ids; element data lives in `map`.
    order: Vec<ElementId>,
    map: HashMap<ElementId, Element>,
}

impl Scene {
    pub fn new() -> Self {
        Scene::default()
    }

    /// Number of elements, including those flagged deleted.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Insert (or replace) an element, appending to the top of the paint order
    /// if new. Returns whether the element was newly inserted.
    pub fn insert(&mut self, element: Element) -> bool {
        let id = element.id.clone();
        let is_new = !self.map.contains_key(&id);
        if is_new {
            self.order.push(id.clone());
        }
        self.map.insert(id, element);
        is_new
    }

    pub fn get(&self, id: &ElementId) -> Option<&Element> {
        self.map.get(id)
    }

    pub fn get_mut(&mut self, id: &ElementId) -> Option<&mut Element> {
        self.map.get_mut(id)
    }

    pub fn contains(&self, id: &ElementId) -> bool {
        self.map.contains_key(id)
    }

    /// Remove an element entirely (hard delete). For soft-delete, set
    /// `is_deleted` on the element instead — that preserves undo history.
    pub fn remove(&mut self, id: &ElementId) -> Option<Element> {
        if let Some(pos) = self.order.iter().position(|e| e == id) {
            self.order.remove(pos);
        }
        self.map.remove(id)
    }

    /// Iterate all elements (including deleted) in paint order.
    pub fn iter(&self) -> impl Iterator<Item = &Element> {
        self.order.iter().filter_map(move |id| self.map.get(id))
    }

    /// Iterate live (non-deleted) elements in paint order.
    pub fn iter_live(&self) -> impl Iterator<Item = &Element> {
        self.iter().filter(|e| !e.is_deleted)
    }

    /// Element ids in paint order.
    pub fn order(&self) -> &[ElementId] {
        &self.order
    }

    /// Bounding box of all live elements' raw boxes. Tight bounds (rotation,
    /// curves) come from the `geometry::bounds` module in Phase 1.
    pub fn rough_bounds(&self) -> Rect {
        self.iter_live()
            .fold(Rect::EMPTY, |acc, e| acc.union(&e.raw_box()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ElementKind;

    fn el(id: &str) -> Element {
        Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            10.0,
            10.0,
            ElementKind::Rectangle,
        )
    }

    #[test]
    fn insert_get_remove() {
        let mut s = Scene::new();
        assert!(s.insert(el("a")));
        assert!(!s.insert(el("a"))); // replace, not new
        assert_eq!(s.len(), 1);
        assert!(s.get(&ElementId::from("a")).is_some());
        assert!(s.remove(&ElementId::from("a")).is_some());
        assert!(s.is_empty());
    }

    #[test]
    fn paint_order_is_insertion_order() {
        let mut s = Scene::new();
        s.insert(el("a"));
        s.insert(el("b"));
        s.insert(el("c"));
        let ids: Vec<_> = s.iter().map(|e| e.id.as_str().to_string()).collect();
        assert_eq!(ids, ["a", "b", "c"]);
    }

    #[test]
    fn iter_live_skips_deleted() {
        let mut s = Scene::new();
        let mut d = el("a");
        d.is_deleted = true;
        s.insert(d);
        s.insert(el("b"));
        let live: Vec<_> = s.iter_live().map(|e| e.id.as_str().to_string()).collect();
        assert_eq!(live, ["b"]);
    }
}
