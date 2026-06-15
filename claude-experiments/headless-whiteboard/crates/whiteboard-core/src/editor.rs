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

use crate::element::GroupId;
use crate::element::{Element, ElementId};
use crate::geometry::{Point, Vec2};
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
    /// The text element currently being edited (typed into), if any. Set when a
    /// text element is placed or a text element is double-clicked; cleared on
    /// commit/escape. While set, `KeyDown` chars are routed into its text.
    editing_text: Option<ElementId>,
    /// Internal clipboard for copy/cut/paste/duplicate.
    clipboard: crate::scene::Clipboard,
    /// Monotonic counter backing deterministic id/group-id generation for
    /// paste/duplicate (no RNG, no clock — stable across runs).
    id_counter: u64,
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
            editing_text: None,
            clipboard: crate::scene::Clipboard::new(),
            id_counter: 0,
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

    /// Place an image element at `(x, y)` with the given size, referencing
    /// `file_id` (the key a backend uses to resolve the decoded pixels — e.g.
    /// `TinySkiaBackend::register_image`). Records an undo point. Returns its id.
    ///
    /// This is the image-placement entry point: an app decodes the file, hands
    /// the pixels to its backend under `file_id`, and calls this to put the image
    /// on the board.
    pub fn add_image(
        &mut self,
        file_id: impl Into<String>,
        x: f64,
        y: f64,
        width: f64,
        height: f64,
    ) -> ElementId {
        use crate::element::{ElementKind, ImageData};
        let file_id = file_id.into();
        let id = ElementId::from(format!("img-{file_id}"));
        let mut data = ImageData::new(file_id);
        data.status = crate::element::ImageStatus::Saved;
        let el = Element::new(id.clone(), 1, x, y, width, height, ElementKind::Image(data));
        self.add_element(el)
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

    // --- Text editing ----------------------------------------------------

    /// The text element currently being edited, if any.
    pub fn editing_text(&self) -> Option<&ElementId> {
        self.editing_text.as_ref()
    }

    /// Whether a text element is currently being typed into.
    pub fn is_editing_text(&self) -> bool {
        self.editing_text.is_some()
    }

    /// Begin editing the given text element (route keystrokes into it). Records
    /// an undo point so the whole edit session is one undoable change. Returns
    /// false if the id is not a live text element.
    pub fn begin_text_edit(&mut self, id: ElementId) -> bool {
        let is_text = self
            .scene
            .get(&id)
            .map(|e| !e.is_deleted && matches!(e.kind, crate::element::ElementKind::Text(_)))
            .unwrap_or(false);
        if !is_text {
            return false;
        }
        // Snapshot once so the whole typing session collapses into one undo step.
        self.history.record(self.scene.clone());
        self.interaction.set_selection([id.clone()]);
        self.editing_text = Some(id);
        true
    }

    /// Insert a character into the element being edited, resizing its box to fit.
    /// No-op when not editing.
    pub fn type_char(&mut self, c: char) -> bool {
        let Some(id) = self.editing_text.clone() else {
            return false;
        };
        self.mutate_text(&id, |t| t.push(c));
        true
    }

    /// Delete the last character of the element being edited.
    pub fn backspace(&mut self) -> bool {
        let Some(id) = self.editing_text.clone() else {
            return false;
        };
        self.mutate_text(&id, |t| {
            t.pop();
        });
        true
    }

    /// Insert a newline into the element being edited.
    pub fn newline(&mut self) -> bool {
        let Some(id) = self.editing_text.clone() else {
            return false;
        };
        self.mutate_text(&id, |t| t.push('\n'));
        true
    }

    /// Finish the current text edit. If the edited text is empty, the element is
    /// removed (Excalidraw discards empty text). Does NOT record undo (the
    /// session's snapshot was taken at [`Editor::begin_text_edit`]).
    pub fn commit_text(&mut self) {
        if let Some(id) = self.editing_text.take() {
            let empty = self
                .scene
                .get(&id)
                .and_then(|e| match &e.kind {
                    crate::element::ElementKind::Text(t) => Some(t.text.is_empty()),
                    _ => None,
                })
                .unwrap_or(true);
            if empty {
                self.scene.remove(&id);
                self.interaction.prune_selection(&self.scene);
            }
        }
    }

    /// Dispatch a key while editing text. Returns whether the scene changed.
    fn handle_text_key(&mut self, key: &crate::interaction::Key) -> bool {
        use crate::interaction::Key;
        match key {
            Key::Char(c) => self.type_char(*c),
            Key::Backspace => self.backspace(),
            Key::Enter => self.newline(),
            Key::Escape => {
                self.commit_text();
                true
            }
            _ => false,
        }
    }

    /// Apply `f` to the editing element's text string, then re-fit its box to the
    /// measured text so the element bounds track the content.
    fn mutate_text(&mut self, id: &ElementId, f: impl FnOnce(&mut String)) {
        use crate::element::ElementKind;
        use crate::text::FontSpec;

        let mut measured: Option<(f64, f64)> = None;
        if let Some(el) = self.scene.get_mut(id) {
            if let ElementKind::Text(data) = &mut el.kind {
                f(&mut data.text);
                data.original_text = Some(data.text.clone());
                // Measure each line; box = widest line × total line height.
                let font = FontSpec {
                    family: data.font_family.clone(),
                    size: data.font_size,
                    line_height: data.line_height,
                };
                let lines: Vec<&str> = if data.text.is_empty() {
                    vec![""]
                } else {
                    data.text.split('\n').collect()
                };
                let width = lines
                    .iter()
                    .map(|l| self.measurer.measure(l, &font).width)
                    .fold(0.0_f64, f64::max);
                let height = lines.len() as f64 * font.line_spacing();
                measured = Some((width, height));
            }
        }
        if let (Some(el), Some((w, h))) = (self.scene.get_mut(id), measured) {
            el.width = w;
            el.height = h;
        }
    }

    // --- Clipboard / duplicate ------------------------------------------

    /// Deterministic id allocator shared by the paste/duplicate/group closures
    /// (a `Cell` so two `FnMut` generators can both pull from one counter).
    fn next_id_cell(counter: &std::cell::Cell<u64>) -> ElementId {
        let n = counter.get();
        counter.set(n + 1);
        ElementId::from(format!("gen-{n}"))
    }

    /// Copy the current selection to the internal clipboard.
    pub fn copy(&mut self) {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        self.clipboard = crate::scene::copy(&self.scene, &ids);
    }

    /// Copy the selection, then soft-delete it (undoable).
    pub fn cut(&mut self) -> bool {
        self.copy();
        self.delete_selection()
    }

    /// Paste the clipboard contents, offset by `offset`, selecting the new
    /// elements. Returns the new element ids. Records one undo entry.
    pub fn paste(&mut self, offset: Vec2) -> Vec<ElementId> {
        if self.clipboard.is_empty() {
            return Vec::new();
        }
        let counter = std::cell::Cell::new(self.id_counter);
        let new_elements = {
            let mut id_gen = || Self::next_id_cell(&counter);
            let mut group_gen = || GroupId::from(Self::next_id_cell(&counter).as_str());
            crate::scene::paste(&self.clipboard, &mut id_gen, &mut group_gen, offset)
        };
        self.id_counter = counter.get();
        if new_elements.is_empty() {
            return Vec::new();
        }
        self.history.record(self.scene.clone());
        let ids: Vec<ElementId> = new_elements.iter().map(|e| e.id.clone()).collect();
        for el in new_elements {
            self.scene.insert(el);
        }
        self.interaction.set_selection(ids.clone());
        ids
    }

    /// Duplicate the selection in place with a small nudge (Excalidraw uses
    /// +10,+10), selecting the duplicates. Records one undo entry.
    pub fn duplicate_selection(&mut self) -> Vec<ElementId> {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        if ids.is_empty() {
            return Vec::new();
        }
        let counter = std::cell::Cell::new(self.id_counter);
        let new_elements = {
            let mut id_gen = || Self::next_id_cell(&counter);
            let mut group_gen = || GroupId::from(Self::next_id_cell(&counter).as_str());
            crate::scene::duplicate(
                &self.scene,
                &ids,
                &mut id_gen,
                &mut group_gen,
                Vec2::new(10.0, 10.0),
            )
        };
        self.id_counter = counter.get();
        if new_elements.is_empty() {
            return Vec::new();
        }
        self.history.record(self.scene.clone());
        let new_ids: Vec<ElementId> = new_elements.iter().map(|e| e.id.clone()).collect();
        for el in new_elements {
            self.scene.insert(el);
        }
        self.interaction.set_selection(new_ids.clone());
        new_ids
    }

    // --- Property mutation / alignment ----------------------------------

    /// Apply a style change (color, fill, stroke width/style, roughness, opacity)
    /// to the current selection. Records one undo entry if anything changed.
    pub fn set_style(&mut self, change: &crate::scene::StyleChange) -> bool {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        if ids.is_empty() {
            return false;
        }
        let before = self.scene.clone();
        if crate::scene::apply_style(&mut self.scene, &ids, change) {
            self.history.record(before);
            true
        } else {
            false
        }
    }

    /// Align the current selection (>=2 elements). Records one undo entry.
    pub fn align(&mut self, how: crate::scene::Align) -> bool {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        let before = self.scene.clone();
        if crate::scene::align(&mut self.scene, &ids, how) {
            self.history.record(before);
            true
        } else {
            false
        }
    }

    /// Distribute the current selection evenly (>=3 elements). Records one undo.
    pub fn distribute(&mut self, how: crate::scene::Distribute) -> bool {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        let before = self.scene.clone();
        if crate::scene::distribute(&mut self.scene, &ids, how) {
            self.history.record(before);
            true
        } else {
            false
        }
    }

    // --- Z-order ---------------------------------------------------------

    /// Bring the current selection to the front (top of the paint order).
    pub fn bring_to_front(&mut self) -> bool {
        self.reorder(|scene, ids| scene.bring_to_front(ids))
    }

    /// Send the current selection to the back (bottom of the paint order).
    pub fn send_to_back(&mut self) -> bool {
        self.reorder(|scene, ids| scene.send_to_back(ids))
    }

    /// Raise the current selection one step up the paint order.
    pub fn raise(&mut self) -> bool {
        self.reorder(|scene, ids| scene.raise(ids))
    }

    /// Lower the current selection one step down the paint order.
    pub fn lower(&mut self) -> bool {
        self.reorder(|scene, ids| scene.lower(ids))
    }

    /// Shared z-order helper: snapshot, apply the reorder to the selection, and
    /// record undo. Returns whether the order changed.
    fn reorder(
        &mut self,
        f: impl FnOnce(&mut Scene, &[ElementId]) -> Result<(), crate::scene::IndexError>,
    ) -> bool {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        if ids.is_empty() {
            return false;
        }
        let before = self.scene.clone();
        let order_before = self.scene.order().to_vec();
        // The z-order operations work on fractional indices; ensure every element
        // has one first (elements added programmatically start with `index: None`).
        if self.scene.assign_initial_indices().is_err() {
            self.scene = before;
            return false;
        }
        if f(&mut self.scene, &ids).is_err() {
            // Restore on error so a failed reorder is a clean no-op.
            self.scene = before;
            return false;
        }
        if self.scene.order() == order_before.as_slice() {
            return false; // no change ⇒ no undo entry
        }
        self.history.record(before);
        true
    }

    // --- Grouping --------------------------------------------------------

    /// Group the current selection under a fresh group id. Returns whether a
    /// group was formed (needs >= 2 elements). Records undo.
    pub fn group_selection(&mut self) -> bool {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        let counter = std::cell::Cell::new(self.id_counter);
        let group_id = GroupId::from(Self::next_id_cell(&counter).as_str());
        self.id_counter = counter.get();
        let before = self.scene.clone();
        if crate::scene::group(&mut self.scene, &ids, group_id) {
            self.history.record(before);
            true
        } else {
            false
        }
    }

    /// Ungroup the current selection's outermost shared group. Returns whether
    /// anything was ungrouped. Records undo.
    pub fn ungroup_selection(&mut self) -> bool {
        let ids: Vec<_> = self.selection().iter().cloned().collect();
        let before = self.scene.clone();
        let removed = crate::scene::ungroup(&mut self.scene, &ids);
        if removed.is_empty() {
            false
        } else {
            self.history.record(before);
            true
        }
    }

    /// Expand a set of ids to include all members of their groups (so clicking
    /// one group member selects the whole group). Used by callers wiring
    /// group-aware click selection.
    pub fn expand_to_groups(&self, ids: &[ElementId]) -> Vec<ElementId> {
        crate::scene::group_members(&self.scene, ids)
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
        // While editing text, route keystrokes into the text element instead of
        // the interaction state machine. A pointer-down ends the edit (click
        // elsewhere to finish), then falls through to normal handling.
        if self.is_editing_text() {
            match &event {
                InputEvent::KeyDown { key, .. } => {
                    let changed = self.handle_text_key(key);
                    return HandleResult {
                        scene_changed: changed,
                        view_changed: changed,
                    };
                }
                InputEvent::KeyUp { .. } => {
                    return HandleResult::default();
                }
                InputEvent::PointerDown { .. } => {
                    // A click ends the edit. Consume this event so it doesn't
                    // immediately place/select something at the click point
                    // (Excalidraw: first click just blurs the text editor).
                    self.commit_text();
                    return HandleResult {
                        scene_changed: true,
                        view_changed: true,
                    };
                }
                _ => {}
            }
        }

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
            Some(commit) => {
                if let Some(before) = self.pending_undo.take() {
                    self.history.record(before);
                }
                // A committed gesture may have moved/resized elements that arrows
                // are bound to; refresh bound arrow endpoints so they follow.
                self.refresh_bindings();
                // Moving an element in/out of a frame updates its membership.
                self.refresh_frame_membership();
                // A newly placed text element enters edit mode automatically so
                // the next keystrokes type into it. (We already recorded undo
                // above, so begin_text_edit must not double-record — set the
                // editing id directly.)
                if let crate::interaction::Commit::Created(id) = commit {
                    if matches!(
                        self.scene.get(id).map(|e| &e.kind),
                        Some(crate::element::ElementKind::Text(_))
                    ) {
                        self.editing_text = Some(id.clone());
                    }
                }
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

    /// After a gesture, reassign frame membership for elements whose containing
    /// frame changed (dragged into or out of a frame). Only applies real
    /// changes, so it is a cheap no-op for the common case.
    fn refresh_frame_membership(&mut self) {
        let ids: Vec<ElementId> = self.scene.iter_live().map(|e| e.id.clone()).collect();
        for id in ids {
            if let Some(new_frame) = crate::scene::assign_frame_membership(&self.scene, &id) {
                if let Some(el) = self.scene.get_mut(&id) {
                    el.frame_id = new_frame;
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

        // Snapping alignment guides: thin pink segments (Excalidraw style) shown
        // while a move snaps to another element. In scene space → map to screen.
        let guides = self.interaction.snap_guides();
        if !guides.is_empty() {
            use crate::geometry::Path;
            use crate::render::{Color, DrawCommand, Paint, Stroke};
            let pink = Color::rgb(255, 0, 200);
            for g in guides {
                let a = to_screen.apply(g.a);
                let b = to_screen.apply(g.b);
                scene.push(DrawCommand::StrokePath {
                    path: Path::polyline(&[a, b]),
                    stroke: Stroke::solid(1.0),
                    paint: Paint::solid(pink),
                });
            }
        }

        // Laser-pointer trail: a transient screen-space polyline (never part of
        // the scene). Drawn last so it sits on top of everything.
        let trail = self.interaction.laser_trail();
        if trail.len() >= 2 {
            use crate::geometry::Path;
            use crate::render::{Color, DrawCommand, Paint, Stroke};
            let screen_pts: Vec<Point> = trail.iter().map(|p| to_screen.apply(*p)).collect();
            let path = Path::polyline(&screen_pts);
            scene.push(DrawCommand::StrokePath {
                path,
                stroke: Stroke::solid(4.0),
                paint: Paint::solid(Color::rgba(255, 32, 32, 200)),
            });
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
