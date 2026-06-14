//! The top-level facade an application holds.
//!
//! [`Editor`] owns the scene, the current tool, the viewport, selection, undo
//! history, and an injected [`TextMeasurer`]. An app feeds it raw
//! [`InputEvent`]s and asks it to [`Editor::render`] a [`RenderScene`] each
//! frame. This is the single entry point that ties every module together.
//!
//! Phase 2 fleshes out [`Editor::handle`] to drive the full interaction state
//! machine. This baseline already supports the core programmatic API (add /
//! select / delete elements, undo/redo, pan/zoom, render) so backends and tests
//! can exercise the whole pipeline immediately.

use crate::element::{Element, ElementId};
use crate::geometry::Point;
use crate::history::History;
use crate::interaction::{InputEvent, Tool, Viewport};
use crate::render::{tessellate, RenderOptions, RenderScene};
use crate::scene::Scene;
use crate::shape::{CleanGenerator, ShapeGenerator};
use crate::text::TextMeasurer;
use std::collections::HashSet;

/// Outcome of handling an input event: did anything change, and does the view
/// need a repaint?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HandleResult {
    /// The scene (elements) changed.
    pub scene_changed: bool,
    /// The view (selection, viewport, tool) changed — repaint needed even if
    /// the scene didn't.
    pub view_changed: bool,
}

impl HandleResult {
    pub fn needs_redraw(&self) -> bool {
        self.scene_changed || self.view_changed
    }
}

/// The whiteboard editor.
///
/// Generic over the [`TextMeasurer`] (injected by the backend) and the
/// [`ShapeGenerator`] (clean vs rough). Defaults to the clean generator; swap in
/// the rough generator once Phase 1 lands it.
pub struct Editor<M: TextMeasurer, G: ShapeGenerator = CleanGenerator> {
    scene: Scene,
    history: History,
    viewport: Viewport,
    tool: Tool,
    selection: HashSet<ElementId>,
    measurer: M,
    generator: G,
}

impl<M: TextMeasurer> Editor<M, CleanGenerator> {
    /// Create an editor with the clean (non-sketchy) generator.
    pub fn new(measurer: M) -> Self {
        Editor::with_generator(measurer, CleanGenerator)
    }
}

impl<M: TextMeasurer, G: ShapeGenerator> Editor<M, G> {
    pub fn with_generator(measurer: M, generator: G) -> Self {
        Editor {
            scene: Scene::new(),
            history: History::default(),
            viewport: Viewport::default(),
            tool: Tool::default(),
            selection: HashSet::new(),
            measurer,
            generator,
        }
    }

    // --- Accessors -------------------------------------------------------

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn viewport(&self) -> Viewport {
        self.viewport
    }

    pub fn set_viewport(&mut self, viewport: Viewport) {
        self.viewport = viewport;
    }

    pub fn tool(&self) -> Tool {
        self.tool
    }

    pub fn set_tool(&mut self, tool: Tool) {
        self.tool = tool;
    }

    pub fn measurer(&self) -> &M {
        &self.measurer
    }

    pub fn selection(&self) -> &HashSet<ElementId> {
        &self.selection
    }

    // --- Programmatic scene API -----------------------------------------

    /// Add an element, recording an undo point. Returns its id.
    pub fn add_element(&mut self, element: Element) -> ElementId {
        let id = element.id.clone();
        self.history.record(self.scene.clone());
        self.scene.insert(element);
        id
    }

    /// Soft-delete the current selection (preserving undo). Returns whether any
    /// element was deleted.
    pub fn delete_selection(&mut self) -> bool {
        if self.selection.is_empty() {
            return false;
        }
        self.history.record(self.scene.clone());
        let ids: Vec<_> = self.selection.iter().cloned().collect();
        let mut changed = false;
        for id in ids {
            if let Some(el) = self.scene.get_mut(&id) {
                el.is_deleted = true;
                changed = true;
            }
        }
        self.selection.clear();
        changed
    }

    /// Replace the selection set.
    pub fn select(&mut self, ids: impl IntoIterator<Item = ElementId>) {
        self.selection = ids.into_iter().collect();
    }

    pub fn clear_selection(&mut self) {
        self.selection.clear();
    }

    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    pub fn undo(&mut self) -> bool {
        if let Some(prev) = self.history.undo(self.scene.clone()) {
            self.scene = prev;
            self.prune_selection();
            true
        } else {
            false
        }
    }

    pub fn redo(&mut self) -> bool {
        if let Some(next) = self.history.redo(self.scene.clone()) {
            self.scene = next;
            self.prune_selection();
            true
        } else {
            false
        }
    }

    /// Drop selected ids that no longer refer to live elements.
    fn prune_selection(&mut self) {
        self.selection
            .retain(|id| self.scene.get(id).map(|e| !e.is_deleted).unwrap_or(false));
    }

    /// Map a screen point to scene coordinates via the viewport.
    pub fn screen_to_scene(&self, p: Point) -> Point {
        self.viewport.screen_to_scene(p)
    }

    // --- Event handling --------------------------------------------------

    /// Handle a raw input event. Phase 2 implements the full per-tool state
    /// machine here; for now it is a no-op that reports no change, so callers
    /// can already wire the event loop without depending on unfinished behavior.
    pub fn handle(&mut self, _event: InputEvent) -> HandleResult {
        HandleResult::default()
    }

    // --- Rendering -------------------------------------------------------

    /// Produce the draw-command list for the current frame.
    pub fn render(&self) -> RenderScene {
        let opts = RenderOptions {
            viewport: self.viewport.scene_to_screen(),
        };
        tessellate(&self.scene, &self.generator, &opts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ElementId, ElementKind};
    use crate::text::MonospaceMeasurer;

    fn editor() -> Editor<MonospaceMeasurer> {
        Editor::new(MonospaceMeasurer::default())
    }

    fn rect(id: &str) -> Element {
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
    fn add_then_undo_then_redo() {
        let mut e = editor();
        e.add_element(rect("a"));
        assert_eq!(e.scene().iter_live().count(), 1);
        assert!(e.undo());
        assert_eq!(e.scene().iter_live().count(), 0);
        assert!(e.redo());
        assert_eq!(e.scene().iter_live().count(), 1);
    }

    #[test]
    fn delete_selection_soft_deletes() {
        let mut e = editor();
        let id = e.add_element(rect("a"));
        e.select([id]);
        assert!(e.delete_selection());
        assert_eq!(e.scene().iter_live().count(), 0);
        // Undo brings it back.
        assert!(e.undo());
        assert_eq!(e.scene().iter_live().count(), 1);
    }

    #[test]
    fn render_produces_commands() {
        let mut e = editor();
        e.add_element(rect("a"));
        let rs = e.render();
        assert!(!rs.is_empty());
    }

    #[test]
    fn undo_prunes_stale_selection() {
        let mut e = editor();
        let id = e.add_element(rect("a"));
        e.select([id.clone()]);
        e.undo(); // element gone
        assert!(!e.selection().contains(&id));
    }
}
