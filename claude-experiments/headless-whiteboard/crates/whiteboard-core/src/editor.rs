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
use crate::interaction::state::InteractionState;
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
    /// The interaction state machine owns the selection set and any in-progress
    /// pointer gesture; the editor drives it from raw events and wraps undo
    /// bookkeeping around its commits.
    interaction: InteractionState,
    /// Set on the pointer-down that begins a gesture: holds the pre-gesture
    /// scene snapshot until the matching commit (then it is pushed to history)
    /// or the gesture is abandoned (then it is dropped).
    pending_undo: Option<Scene>,
    measurer: M,
    generator: G,
}

impl<M: TextMeasurer> Editor<M, CleanGenerator> {
    /// Create an editor with the clean (non-sketchy) generator.
    pub fn new(measurer: M) -> Self {
        Editor::with_generator(measurer, CleanGenerator)
    }
}

impl<M: TextMeasurer> Editor<M, crate::shape::RoughGenerator> {
    /// Create an editor with the hand-drawn ("sketchy") generator — the default
    /// Excalidraw look. Each element's `seed`/`roughness` drives the sketch, so
    /// rendering stays deterministic.
    pub fn new_rough(measurer: M) -> Self {
        Editor::with_generator(measurer, crate::shape::RoughGenerator::new())
    }
}

impl<M: TextMeasurer, G: ShapeGenerator> Editor<M, G> {
    pub fn with_generator(measurer: M, generator: G) -> Self {
        Editor {
            scene: Scene::new(),
            history: History::default(),
            viewport: Viewport::default(),
            tool: Tool::default(),
            interaction: InteractionState::default(),
            pending_undo: None,
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
        self.interaction.selection()
    }

    /// Mutable access to the interaction state machine (tools, gestures, handle
    /// layout). Backends use this for handle hit-testing / overlay rendering.
    pub fn interaction(&self) -> &InteractionState {
        &self.interaction
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
        if self.selection().is_empty() {
            return false;
        }
        self.history.record(self.scene.clone());
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        let mut changed = false;
        for id in ids {
            if let Some(el) = self.scene.get_mut(&id) {
                el.is_deleted = true;
                changed = true;
            }
        }
        self.interaction.clear_selection();
        changed
    }

    /// Replace the selection set.
    pub fn select(&mut self, ids: impl IntoIterator<Item = ElementId>) {
        self.interaction.set_selection(ids);
    }

    pub fn clear_selection(&mut self) {
        self.interaction.clear_selection();
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
        self.interaction.prune_selection(&self.scene);
    }

    /// Map a screen point to scene coordinates via the viewport.
    pub fn screen_to_scene(&self, p: Point) -> Point {
        self.viewport.screen_to_scene(p)
    }

    // --- Event handling --------------------------------------------------

    /// Handle a raw input event, driving the interaction state machine and
    /// wrapping undo bookkeeping around any committed gesture.
    ///
    /// The contract with [`InteractionState`]: when it reports `begins_undo` we
    /// snapshot the pre-gesture scene; when it reports a `commit` we push that
    /// snapshot to history; if a gesture ends without a commit (e.g. a bare
    /// click that created nothing) we discard the snapshot so no empty undo
    /// entry is recorded.
    pub fn handle(&mut self, event: InputEvent) -> HandleResult {
        let is_pointer_up = matches!(event, InputEvent::PointerUp { .. });
        // Snapshot the scene *before* the state machine touches it. A gesture
        // that mutates the scene (create/move/resize/rotate) begins on a
        // pointer-down; the create gesture inserts its element during that same
        // event, so capturing here — before `handle` — yields the true
        // pre-gesture scene to restore on undo.
        let before = self.scene.clone();

        let result = self
            .interaction
            .handle(&mut self.scene, &mut self.viewport, self.tool, event);

        if result.begins_undo {
            self.pending_undo = Some(before);
        }

        match &result.commit {
            Some(_) => {
                if let Some(before) = self.pending_undo.take() {
                    self.history.record(before);
                }
                // A committed gesture may have moved/resized elements that arrows
                // are bound to; refresh bound arrow endpoints so they follow.
                self.refresh_bindings();
            }
            None => {
                // A pointer-up that ends a gesture without committing (e.g. a
                // bare click that created nothing) discards the snapshot so no
                // empty undo entry is recorded.
                if is_pointer_up {
                    self.pending_undo = None;
                }
            }
        }

        HandleResult {
            scene_changed: result.scene_changed,
            view_changed: result.view_changed,
        }
    }

    /// Recompute the endpoints of every bound arrow from its targets' current
    /// positions, so arrows stay attached when shapes move/resize. Cheap sweep:
    /// only arrows that actually carry a binding are recomputed.
    fn refresh_bindings(&mut self) {
        use crate::element::{apply_bound_endpoints, update_bound_arrow, ElementKind};

        // Collect arrow ids that have at least one binding.
        let bound_arrows: Vec<ElementId> = self
            .scene
            .iter_live()
            .filter(|e| match &e.kind {
                ElementKind::Arrow(d) | ElementKind::Line(d) => {
                    d.start_binding.is_some() || d.end_binding.is_some()
                }
                _ => false,
            })
            .map(|e| e.id.clone())
            .collect();

        for id in bound_arrows {
            // Recompute against the (immutable) scene, then apply to the arrow.
            let Some(endpoints) = update_bound_arrow(&self.scene, &id) else {
                continue;
            };
            if let Some(arrow) = self.scene.get_mut(&id) {
                let origin = Point::new(arrow.x, arrow.y);
                if let ElementKind::Arrow(d) | ElementKind::Line(d) = &mut arrow.kind {
                    apply_bound_endpoints(d, origin, endpoints);
                }
            }
        }
    }

    // --- Rendering -------------------------------------------------------

    /// Produce the draw-command list for the current frame.
    pub fn render(&self) -> RenderScene {
        let opts = RenderOptions {
            viewport: self.viewport.scene_to_screen(),
        };
        tessellate(&self.scene, &self.generator, &self.measurer, &opts)
    }

    /// Render the scene plus the selection overlay (bounding box, resize/rotation
    /// handles, and the active marquee) on top, using the default overlay style.
    ///
    /// The overlay commands are in **screen** space and are appended after the
    /// viewport-transformed scene, so a backend draws them as on-screen UI.
    pub fn render_with_overlay(&self) -> RenderScene {
        self.render_with_overlay_style(&crate::render::OverlayStyle::default())
    }

    /// Like [`Editor::render_with_overlay`] but with a caller-supplied style.
    pub fn render_with_overlay_style(&self, style: &crate::render::OverlayStyle) -> RenderScene {
        let mut scene = self.render();

        let layout = self.interaction.handle_layout(&self.scene, &self.viewport);
        // Map the scene-space marquee into screen space for the overlay.
        let to_screen = self.viewport.scene_to_screen();
        let marquee = self
            .interaction
            .active_marquee()
            .map(|r| to_screen.apply_rect_bounds(&r));

        if layout.is_some() || marquee.is_some() {
            let overlay = crate::render::selection_overlay(layout.as_ref(), marquee, style);
            scene.commands.extend(overlay.commands);
            scene.bounds = scene.bounds.union(&overlay.bounds);
        }
        scene
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
