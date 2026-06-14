//! Undo / redo.
//!
//! History records reversible changes to the scene. Phase 1 wires this to the
//! actual scene-mutation path (capturing element snapshots around each
//! interaction) and adds change coalescing. This file provides a working,
//! tested snapshot-based stack that the editor can already use: push a snapshot
//! before a mutation, then undo/redo swaps whole [`Scene`] states.
//!
//! Snapshot-of-whole-scene is simple and correct; a delta-based optimization can
//! replace the internals later without changing this API.

use crate::scene::Scene;

/// Bounded undo/redo stack over full scene snapshots.
#[derive(Debug, Clone)]
pub struct History {
    past: Vec<Scene>,
    future: Vec<Scene>,
    limit: usize,
}

impl Default for History {
    fn default() -> Self {
        History::new(200)
    }
}

impl History {
    pub fn new(limit: usize) -> Self {
        History {
            past: Vec::new(),
            future: Vec::new(),
            limit: limit.max(1),
        }
    }

    /// Record `before` as a restore point prior to a mutation. Clears the redo
    /// stack, since a new edit invalidates any redone-away future.
    pub fn record(&mut self, before: Scene) {
        self.future.clear();
        self.past.push(before);
        if self.past.len() > self.limit {
            self.past.remove(0);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.past.is_empty()
    }

    pub fn can_redo(&self) -> bool {
        !self.future.is_empty()
    }

    /// Undo: given the `current` scene, return the previous one and stash
    /// `current` for redo. Returns `None` if nothing to undo.
    pub fn undo(&mut self, current: Scene) -> Option<Scene> {
        let prev = self.past.pop()?;
        self.future.push(current);
        Some(prev)
    }

    /// Redo: inverse of [`History::undo`].
    pub fn redo(&mut self, current: Scene) -> Option<Scene> {
        let next = self.future.pop()?;
        self.past.push(current);
        Some(next)
    }

    pub fn clear(&mut self) {
        self.past.clear();
        self.future.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Element, ElementId, ElementKind};

    fn scene_with(id: &str) -> Scene {
        let mut s = Scene::new();
        s.insert(Element::new(
            ElementId::from(id),
            1,
            0.0,
            0.0,
            1.0,
            1.0,
            ElementKind::Rectangle,
        ));
        s
    }

    #[test]
    fn undo_redo_round_trip() {
        let mut h = History::default();
        let s0 = Scene::new();
        let s1 = scene_with("a");

        h.record(s0.clone()); // about to go from s0 -> s1
        assert!(h.can_undo());

        let undone = h.undo(s1.clone()).unwrap();
        assert_eq!(undone.len(), 0); // back to s0
        assert!(h.can_redo());

        let redone = h.redo(undone).unwrap();
        assert_eq!(redone.len(), 1); // forward to s1
    }

    #[test]
    fn new_edit_clears_redo() {
        let mut h = History::default();
        h.record(Scene::new());
        let _ = h.undo(scene_with("a"));
        assert!(h.can_redo());
        h.record(scene_with("a")); // new edit
        assert!(!h.can_redo());
    }

    #[test]
    fn limit_is_enforced() {
        let mut h = History::new(2);
        for _ in 0..5 {
            h.record(Scene::new());
        }
        assert_eq!(h.past.len(), 2);
    }
}
