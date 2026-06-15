//! The interaction state machine.
//!
//! Reimplemented from Excalidraw's pointer pipeline (the `handleCanvasPointerDown`
//! / `onPointerMoveFromPointerDownHandler` / `onPointerUp` trio in
//! `packages/excalidraw/components/App.tsx`, plus `resizeElements.ts` and
//! `dragElements.ts`). [`InteractionState`] consumes [`InputEvent`]s and mutates a
//! [`Scene`] + [`Viewport`] for the current [`Tool`]: drag-to-create, click /
//! shift-click / marquee selection, drag-to-move, 8-handle resize, rotate, pan,
//! and wheel zoom.
//!
//! It is deliberately *self-contained*: it owns the selection set and an
//! in-progress gesture, and reports — via [`InteractionResult`] — exactly what
//! changed so [`crate::editor::Editor`] can record an undo snapshot at the right
//! moment (it captures the pre-gesture scene on pointer-down and commits on up).
//!
//! Hit-testing currently uses each element's axis-aligned [`Element::raw_box`]
//! (correct for unrotated boxes). TODO: switch to `geometry::hit::hit_test` once
//! that module lands (built in parallel) for precise per-shape and rotated hits.

use crate::element::{Element, ElementId};
use crate::geometry::{point_rotate_rads, Point, Rect, Vec2};
use crate::interaction::handles::{Handle, HandleLayout};
use crate::interaction::tools::{self, CreateKind};
use crate::interaction::{InputEvent, Key, Modifiers, PointerButton, Tool, Viewport};
use crate::rough::RoughRng;
use crate::scene::Scene;
use std::collections::HashSet;

/// Below this scene-space drag distance a pointer-down/up is treated as a click,
/// not a drag (prevents accidental 1px marquees / moves).
const CLICK_SLOP: f64 = 2.0;

/// How a single `handle` call changed the world. The editor turns this into a
/// [`crate::editor::HandleResult`] and decides undo bookkeeping.
#[derive(Debug, Clone, PartialEq)]
pub struct InteractionResult {
    /// The scene's elements were mutated (created / moved / resized / rotated).
    pub scene_changed: bool,
    /// Selection, hover, viewport or in-progress-gesture state changed.
    pub view_changed: bool,
    /// A discrete change worth an undo entry just *completed* (e.g. pointer-up
    /// ending a create/move/resize/rotate). The editor should have captured a
    /// snapshot when [`InteractionResult::begins_undo`] was set on the matching
    /// pointer-down.
    pub commit: Option<Commit>,
    /// This event *started* a scene-mutating gesture; the editor should snapshot
    /// the scene now so the eventual commit is undoable.
    pub begins_undo: bool,
}

impl InteractionResult {
    fn none() -> Self {
        InteractionResult {
            scene_changed: false,
            view_changed: false,
            commit: None,
            begins_undo: false,
        }
    }
    fn view() -> Self {
        InteractionResult {
            view_changed: true,
            ..Self::none()
        }
    }
    pub fn needs_redraw(&self) -> bool {
        self.scene_changed || self.view_changed
    }
}

/// A completed, undoable change, naming what happened and which elements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Commit {
    /// A new element was created and committed. (id of the new element)
    Created(ElementId),
    /// One or more elements were moved.
    Moved(Vec<ElementId>),
    /// The selection was resized.
    Resized(Vec<ElementId>),
    /// The selection was rotated.
    Rotated(Vec<ElementId>),
}

/// The gesture currently in progress between a pointer-down and pointer-up.
#[derive(Debug, Clone)]
enum Gesture {
    /// Drawing a new element by dragging.
    Creating {
        id: ElementId,
        start: Point,
        kind: CreateKind,
        /// Whether at least one move has actually changed the element (so a bare
        /// click with a creation tool doesn't leave a zero-size element).
        moved: bool,
    },
    /// Dragging the selection to move it. Stores each element's original box
    /// origin so movement is computed from the gesture start (no drift).
    Moving {
        start: Point,
        origins: Vec<(ElementId, Point)>,
        moved: bool,
    },
    /// Box-select marquee from an empty-space drag.
    Marquee {
        start: Point,
        /// Current pointer position (scene coords); with `start` defines the
        /// marquee rect. Updated on each pointer-move so the overlay can draw it.
        current: Point,
        /// Selection present before the marquee (for additive shift-drag).
        base: HashSet<ElementId>,
        additive: bool,
    },
    /// Resizing the selection via a handle.
    Resizing {
        handle: Handle,
        start_bbox: Rect,
        /// Original (id, box) of each selected element at gesture start.
        originals: Vec<(ElementId, Rect)>,
        moved: bool,
    },
    /// Rotating the selection via the rotation handle.
    Rotating {
        pivot: Point,
        start_angle: f64,
        /// Original (id, angle, center) of each element.
        originals: Vec<(ElementId, f64, Point)>,
        moved: bool,
    },
    /// Panning the viewport (space-drag / middle-button / Pan tool).
    Panning { last_screen: Point },
}

/// Owns selection + the in-progress gesture and drives all interaction.
#[derive(Debug, Clone)]
pub struct InteractionState {
    selection: HashSet<ElementId>,
    gesture: Option<Gesture>,
    /// True while the space bar is held (temporary pan override).
    space_down: bool,
    /// Deterministic id counter; combined with `id_prefix` to form element ids.
    next_id: u64,
    id_prefix: String,
    /// RNG for per-element seeds (kept off any global/OS source).
    seed_rng: RoughRng,
}

impl Default for InteractionState {
    fn default() -> Self {
        InteractionState::new("el", 0x1234_5678)
    }
}

impl InteractionState {
    /// Create a state machine. `id_prefix` namespaces generated ids and
    /// `seed_base` seeds the deterministic per-element RNG, so tests and
    /// collaborative sessions can be reproducible.
    pub fn new(id_prefix: impl Into<String>, seed_base: u32) -> Self {
        InteractionState {
            selection: HashSet::new(),
            gesture: None,
            space_down: false,
            next_id: 0,
            id_prefix: id_prefix.into(),
            seed_rng: RoughRng::new(seed_base),
        }
    }

    // --- Accessors -------------------------------------------------------

    pub fn selection(&self) -> &HashSet<ElementId> {
        &self.selection
    }

    pub fn set_selection(&mut self, ids: impl IntoIterator<Item = ElementId>) {
        self.selection = ids.into_iter().collect();
    }

    pub fn clear_selection(&mut self) {
        self.selection.clear();
    }

    /// Whether a gesture is currently in progress.
    pub fn is_interacting(&self) -> bool {
        self.gesture.is_some()
    }

    /// The active marquee rectangle in **scene** coordinates, if a box-select
    /// drag is in progress. Returns `None` otherwise. The editor maps this to
    /// screen space for the selection overlay.
    pub fn active_marquee(&self) -> Option<Rect> {
        match &self.gesture {
            Some(Gesture::Marquee { start, current, .. }) => {
                Some(Rect::from_corners(*start, *current))
            }
            _ => None,
        }
    }

    /// Drop selected ids that no longer point at live elements.
    pub fn prune_selection(&mut self, scene: &Scene) {
        self.selection
            .retain(|id| scene.get(id).map(|e| !e.is_deleted).unwrap_or(false));
    }

    /// The selection bounding box (union of selected raw boxes), or `None` if the
    /// selection is empty. TODO: use tight bounds once `geometry::bounds` lands.
    pub fn selection_bbox(&self, scene: &Scene) -> Option<Rect> {
        let mut acc = Rect::EMPTY;
        let mut any = false;
        for id in &self.selection {
            if let Some(el) = scene.get(id) {
                if !el.is_deleted {
                    acc = acc.union(&rotated_bounds(el));
                    any = true;
                }
            }
        }
        any.then_some(acc)
    }

    /// Compute the handle layout for the current selection, if any.
    pub fn handle_layout(&self, scene: &Scene, vp: &Viewport) -> Option<HandleLayout> {
        let bbox = self.selection_bbox(scene)?;
        // A single rotated element exposes its own angle; multi-select uses 0.
        let angle = self.single_selection_angle(scene).unwrap_or(0.0);
        Some(HandleLayout::new(bbox, angle, vp))
    }

    fn single_selection_angle(&self, scene: &Scene) -> Option<f64> {
        if self.selection.len() != 1 {
            return None;
        }
        let id = self.selection.iter().next()?;
        scene.get(id).map(|e| e.angle)
    }

    // --- Entry point -----------------------------------------------------

    /// Process one input event, mutating `scene` and `viewport`. Returns what
    /// changed so the caller can drive undo and repaint.
    pub fn handle(
        &mut self,
        scene: &mut Scene,
        viewport: &mut Viewport,
        tool: Tool,
        event: InputEvent,
    ) -> InteractionResult {
        match event {
            InputEvent::KeyDown { key, .. } => self.on_key(key, true),
            InputEvent::KeyUp { key, .. } => self.on_key(key, false),
            InputEvent::Wheel {
                delta, pos, mods, ..
            } => self.on_wheel(viewport, delta, pos, mods),
            InputEvent::PointerDown { pos, button, mods } => {
                self.on_pointer_down(scene, viewport, tool, pos, button, mods)
            }
            InputEvent::PointerMove { pos, mods } => {
                self.on_pointer_move(scene, viewport, pos, mods)
            }
            InputEvent::PointerUp { pos, button, mods } => {
                self.on_pointer_up(scene, viewport, pos, button, mods)
            }
        }
    }

    // --- Keyboard --------------------------------------------------------

    fn on_key(&mut self, key: Key, down: bool) -> InteractionResult {
        if key == Key::Char(' ') {
            let changed = self.space_down != down;
            self.space_down = down;
            return if changed {
                InteractionResult::view()
            } else {
                InteractionResult::none()
            };
        }
        if down && key == Key::Escape && self.gesture.is_some() {
            // Cancelling an in-progress gesture is left to the editor (it owns
            // the pre-gesture snapshot); here we just drop the gesture.
            self.gesture = None;
            return InteractionResult::view();
        }
        InteractionResult::none()
    }

    // --- Wheel / zoom ----------------------------------------------------

    fn on_wheel(
        &mut self,
        vp: &mut Viewport,
        delta: Vec2,
        pos: Point,
        mods: Modifiers,
    ) -> InteractionResult {
        if mods.command() {
            // Ctrl/Cmd + wheel = zoom toward the cursor.
            let factor = (-delta.y * 0.01).exp();
            vp.zoom_to(vp.zoom * factor, pos);
        } else {
            // Plain wheel = pan. Scene scroll is in scene units, so divide by zoom.
            vp.scroll = Vec2::new(
                vp.scroll.x + delta.x / vp.zoom,
                vp.scroll.y + delta.y / vp.zoom,
            );
        }
        InteractionResult::view()
    }

    // --- Pointer down ----------------------------------------------------

    fn on_pointer_down(
        &mut self,
        scene: &mut Scene,
        vp: &mut Viewport,
        tool: Tool,
        screen: Point,
        button: PointerButton,
        mods: Modifiers,
    ) -> InteractionResult {
        // Pan takes precedence: middle button, Pan tool, or space held.
        if button == PointerButton::Middle || tool == Tool::Pan || self.space_down {
            self.gesture = Some(Gesture::Panning {
                last_screen: screen,
            });
            return InteractionResult::view();
        }

        let scene_pt = vp.screen_to_scene(screen);

        // Creation tools: start a new element.
        if let Some(kind) = tools::create_kind(tool) {
            return self.begin_create(scene, tool, kind, scene_pt);
        }

        // Select tool: first test selection handles, then element hit.
        if tool == Tool::Select {
            // Resize / rotate handles (only when something is selected).
            if let Some(layout) = self.handle_layout(scene, vp) {
                if let Some(h) = layout.hit(screen) {
                    return self.begin_handle(scene, h, &layout, vp);
                }
            }
            // Element under the cursor?
            if let Some(id) = topmost_at(scene, scene_pt) {
                return self.begin_click_or_move(scene, id, scene_pt, mods);
            }
            // Empty space: start a marquee.
            return self.begin_marquee(scene_pt, mods);
        }

        // Non-interactive tools (Text/Image/Frame/Eraser/Laser) are out of scope
        // for this module; report no change rather than faking behavior.
        InteractionResult::none()
    }

    fn begin_create(
        &mut self,
        scene: &mut Scene,
        tool: Tool,
        kind: CreateKind,
        start: Point,
    ) -> InteractionResult {
        let id = self.alloc_id();
        let seed = self.alloc_seed();
        let el = tools::begin_element(tool, id.clone(), seed, start)
            .expect("create_kind matched, so begin_element must produce an element");
        scene.insert(el);
        self.selection = std::iter::once(id.clone()).collect();
        self.gesture = Some(Gesture::Creating {
            id,
            start,
            kind,
            moved: false,
        });
        InteractionResult {
            scene_changed: true,
            view_changed: true,
            commit: None,
            begins_undo: true,
        }
    }

    fn begin_handle(
        &mut self,
        scene: &Scene,
        handle: Handle,
        layout: &HandleLayout,
        _vp: &Viewport,
    ) -> InteractionResult {
        if handle.is_rotation() {
            let originals: Vec<(ElementId, f64, Point)> = self
                .selection
                .iter()
                .filter_map(|id| scene.get(id).map(|e| (id.clone(), e.angle, e.center())))
                .collect();
            let pivot = self
                .selection_bbox(scene)
                .map(|b| b.center())
                .unwrap_or(Point::ORIGIN);
            // Reference angle: the direction from the pivot to the rotation
            // handle at grab time, in scene space. The handle sits along the
            // selection's local "up" vector; for a selection rotated by `angle`
            // that direction is `angle - PI/2` (up is -90deg from +x).
            let sel_angle = self.single_selection_angle(scene).unwrap_or(0.0);
            let start_angle = sel_angle - std::f64::consts::FRAC_PI_2;
            self.gesture = Some(Gesture::Rotating {
                pivot,
                start_angle,
                originals,
                moved: false,
            });
            let _ = layout;
            InteractionResult {
                view_changed: true,
                begins_undo: true,
                ..InteractionResult::none()
            }
        } else {
            let start_bbox = self.selection_bbox(scene).unwrap_or(Rect::EMPTY);
            let originals: Vec<(ElementId, Rect)> = self
                .selection
                .iter()
                .filter_map(|id| scene.get(id).map(|e| (id.clone(), e.raw_box())))
                .collect();
            self.gesture = Some(Gesture::Resizing {
                handle,
                start_bbox,
                originals,
                moved: false,
            });
            InteractionResult {
                view_changed: true,
                begins_undo: true,
                ..InteractionResult::none()
            }
        }
    }

    fn begin_click_or_move(
        &mut self,
        scene: &Scene,
        id: ElementId,
        start: Point,
        mods: Modifiers,
    ) -> InteractionResult {
        if mods.shift {
            // Shift-click toggles membership; doesn't start a move.
            if !self.selection.remove(&id) {
                self.selection.insert(id);
            }
            return InteractionResult::view();
        }
        // Plain click on an unselected element selects just it; on an already
        // selected element keeps the (possibly multi) selection. Either way we
        // arm a potential move drag.
        if !self.selection.contains(&id) {
            self.selection = std::iter::once(id.clone()).collect();
        }
        let origins: Vec<(ElementId, Point)> = self
            .selection
            .iter()
            .filter_map(|sid| scene.get(sid).map(|e| (sid.clone(), Point::new(e.x, e.y))))
            .collect();
        self.gesture = Some(Gesture::Moving {
            start,
            origins,
            moved: false,
        });
        InteractionResult {
            view_changed: true,
            begins_undo: true,
            ..InteractionResult::none()
        }
    }

    fn begin_marquee(&mut self, start: Point, mods: Modifiers) -> InteractionResult {
        let additive = mods.shift;
        let base = if additive {
            self.selection.clone()
        } else {
            HashSet::new()
        };
        if !additive {
            self.selection.clear();
        }
        self.gesture = Some(Gesture::Marquee {
            start,
            current: start,
            base,
            additive,
        });
        InteractionResult::view()
    }

    // --- Pointer move ----------------------------------------------------

    fn on_pointer_move(
        &mut self,
        scene: &mut Scene,
        vp: &mut Viewport,
        screen: Point,
        mods: Modifiers,
    ) -> InteractionResult {
        let scene_pt = vp.screen_to_scene(screen);
        let Some(gesture) = self.gesture.take() else {
            return InteractionResult::none();
        };
        let (next, result) = self.advance_gesture(scene, vp, gesture, scene_pt, screen, mods);
        self.gesture = next;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_gesture(
        &mut self,
        scene: &mut Scene,
        vp: &mut Viewport,
        gesture: Gesture,
        scene_pt: Point,
        screen: Point,
        mods: Modifiers,
    ) -> (Option<Gesture>, InteractionResult) {
        match gesture {
            Gesture::Creating {
                id,
                start,
                kind,
                moved: _,
            } => {
                if let Some(el) = scene.get_mut(&id) {
                    match kind {
                        CreateKind::Box => tools::update_box(el, start, scene_pt, mods.shift),
                        CreateKind::Linear => tools::update_linear(el, start, scene_pt, mods.shift),
                        CreateKind::Freedraw => tools::push_freedraw(el, start, scene_pt),
                    }
                }
                (
                    Some(Gesture::Creating {
                        id,
                        start,
                        kind,
                        moved: true,
                    }),
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        ..InteractionResult::none()
                    },
                )
            }
            Gesture::Moving {
                start,
                origins,
                moved: _,
            } => {
                let d = Vec2::new(scene_pt.x - start.x, scene_pt.y - start.y);
                let moved = d.length() >= CLICK_SLOP;
                if moved {
                    for (id, origin) in &origins {
                        if let Some(el) = scene.get_mut(id) {
                            el.x = origin.x + d.x;
                            el.y = origin.y + d.y;
                        }
                    }
                }
                (
                    Some(Gesture::Moving {
                        start,
                        origins,
                        moved,
                    }),
                    InteractionResult {
                        scene_changed: moved,
                        view_changed: true,
                        ..InteractionResult::none()
                    },
                )
            }
            Gesture::Marquee {
                start,
                current: _,
                base,
                additive,
            } => {
                let marquee = Rect::from_corners(start, scene_pt);
                let mut sel = base.clone();
                for el in scene.iter_live() {
                    if marquee.contains_rect(&el.raw_box()) {
                        sel.insert(el.id.clone());
                    }
                }
                self.selection = sel;
                (
                    Some(Gesture::Marquee {
                        start,
                        current: scene_pt,
                        base,
                        additive,
                    }),
                    InteractionResult::view(),
                )
            }
            Gesture::Resizing {
                handle,
                start_bbox,
                originals,
                moved: _,
            } => {
                let scale = resize_scale(handle, start_bbox, scene_pt, mods.shift);
                for (id, orig) in &originals {
                    if let Some(el) = scene.get_mut(id) {
                        let nb = apply_scale(*orig, start_bbox, handle, scale);
                        el.x = nb.x;
                        el.y = nb.y;
                        el.width = nb.width;
                        el.height = nb.height;
                    }
                }
                (
                    Some(Gesture::Resizing {
                        handle,
                        start_bbox,
                        originals,
                        moved: true,
                    }),
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        ..InteractionResult::none()
                    },
                )
            }
            Gesture::Rotating {
                pivot,
                start_angle,
                originals,
                moved,
            } => {
                let cursor_angle = (scene_pt.y - pivot.y).atan2(scene_pt.x - pivot.x);
                // `start_angle` was captured at grab time (the pivot->handle
                // direction), so rotation tracks the cursor with no initial jump.
                let _ = moved;
                let delta = cursor_angle - start_angle;
                for (id, orig_angle, center) in &originals {
                    if let Some(el) = scene.get_mut(id) {
                        el.angle = crate::geometry::normalize_angle(orig_angle + delta);
                        // Rotate each element's center around the shared pivot so
                        // multi-select rotates as a rigid body.
                        let new_center = point_rotate_rads(*center, pivot, delta);
                        el.x = new_center.x - el.width / 2.0;
                        el.y = new_center.y - el.height / 2.0;
                    }
                }
                let _ = screen;
                let _ = vp;
                (
                    Some(Gesture::Rotating {
                        pivot,
                        start_angle,
                        originals,
                        moved: true,
                    }),
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        ..InteractionResult::none()
                    },
                )
            }
            Gesture::Panning { last_screen } => {
                let dx = (screen.x - last_screen.x) / vp.zoom;
                let dy = (screen.y - last_screen.y) / vp.zoom;
                vp.scroll = Vec2::new(vp.scroll.x - dx, vp.scroll.y - dy);
                (
                    Some(Gesture::Panning {
                        last_screen: screen,
                    }),
                    InteractionResult::view(),
                )
            }
        }
    }

    // --- Pointer up ------------------------------------------------------

    fn on_pointer_up(
        &mut self,
        scene: &mut Scene,
        _vp: &mut Viewport,
        _screen: Point,
        _button: PointerButton,
        _mods: Modifiers,
    ) -> InteractionResult {
        let Some(gesture) = self.gesture.take() else {
            return InteractionResult::none();
        };
        match gesture {
            Gesture::Creating {
                id, moved, kind, ..
            } => {
                // A bare click with a creation tool (no drag) leaves a degenerate
                // element. Drop it instead of committing junk — except freedraw,
                // where a tap is a legitimate dot.
                let keep = moved || kind == CreateKind::Freedraw;
                if keep {
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        commit: Some(Commit::Created(id)),
                        ..InteractionResult::none()
                    }
                } else {
                    scene.remove(&id);
                    self.selection.remove(&id);
                    // Undo was opened on down; signal a (no-op) scene change so the
                    // editor can discard its snapshot.
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        commit: None,
                        ..InteractionResult::none()
                    }
                }
            }
            Gesture::Moving { origins, moved, .. } => {
                if moved {
                    let ids = origins.into_iter().map(|(id, _)| id).collect();
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        commit: Some(Commit::Moved(ids)),
                        ..InteractionResult::none()
                    }
                } else {
                    // A click that didn't drag: the selection set is the result.
                    InteractionResult {
                        commit: None,
                        ..InteractionResult::view()
                    }
                }
            }
            Gesture::Marquee { .. } => InteractionResult::view(),
            Gesture::Resizing {
                originals, moved, ..
            } => {
                if moved {
                    let ids = originals.into_iter().map(|(id, _)| id).collect();
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        commit: Some(Commit::Resized(ids)),
                        ..InteractionResult::none()
                    }
                } else {
                    InteractionResult::view()
                }
            }
            Gesture::Rotating {
                originals, moved, ..
            } => {
                if moved {
                    let ids = originals.into_iter().map(|(id, _, _)| id).collect();
                    InteractionResult {
                        scene_changed: true,
                        view_changed: true,
                        commit: Some(Commit::Rotated(ids)),
                        ..InteractionResult::none()
                    }
                } else {
                    InteractionResult::view()
                }
            }
            Gesture::Panning { .. } => InteractionResult::view(),
        }
    }

    // --- Id / seed allocation -------------------------------------------

    fn alloc_id(&mut self) -> ElementId {
        let id = ElementId::new(format!("{}-{}", self.id_prefix, self.next_id));
        self.next_id += 1;
        id
    }

    fn alloc_seed(&mut self) -> u32 {
        // Pull a deterministic 31-bit seed from the reused Rough RNG.
        (self.seed_rng.next_f64() * (i32::MAX as f64)) as u32
    }
}

// --- Free helpers --------------------------------------------------------

/// Topmost live element whose (unrotated) raw box contains `scene_pt`.
///
/// TODO: replace with `geometry::hit::hit_test` for precise per-shape +
/// rotation-aware hits once that module exists. For axis-aligned boxes this is
/// already correct; for rotated elements it tests the unrotated box, which the
/// hit module will refine.
fn topmost_at(scene: &Scene, scene_pt: Point) -> Option<ElementId> {
    // Paint order is bottom-first, so iterate ids in reverse for top-first.
    for id in scene.order().iter().rev() {
        if let Some(el) = scene.get(id) {
            if !el.is_deleted && hit_element(el, scene_pt) {
                return Some(id.clone());
            }
        }
    }
    None
}

/// Hit-test a single element. Rotated elements: inverse-rotate the point about
/// the element center, then test the raw box (correct for boxes; the dedicated
/// hit module will add per-shape precision).
fn hit_element(el: &Element, scene_pt: Point) -> bool {
    let local = if el.angle != 0.0 {
        point_rotate_rads(scene_pt, el.center(), -el.angle)
    } else {
        scene_pt
    };
    el.raw_box().contains(local)
}

/// The axis-aligned bounds of an element accounting for its rotation (so the
/// selection box wraps a rotated element). Uses the raw box corners.
fn rotated_bounds(el: &Element) -> Rect {
    if el.angle == 0.0 {
        return el.raw_box();
    }
    let c = el.center();
    Rect::bounding(
        el.raw_box()
            .corners()
            .into_iter()
            .map(|p| point_rotate_rads(p, c, el.angle)),
    )
}

/// Scale factors `(sx, sy)` for a resize, derived from how far `cursor` is from
/// the handle's fixed anchor (the opposite corner/edge of the box). With
/// `keep_aspect` the smaller axis follows the larger (Excalidraw shift-resize).
fn resize_scale(handle: Handle, bbox: Rect, cursor: Point, keep_aspect: bool) -> (f64, f64) {
    let mut sx = 1.0;
    let mut sy = 1.0;
    if handle.affects_left() && bbox.width != 0.0 {
        // anchor = right edge
        sx = (bbox.max_x() - cursor.x) / bbox.width;
    } else if handle.affects_right() && bbox.width != 0.0 {
        // anchor = left edge
        sx = (cursor.x - bbox.min_x()) / bbox.width;
    }
    if handle.affects_top() && bbox.height != 0.0 {
        sy = (bbox.max_y() - cursor.y) / bbox.height;
    } else if handle.affects_bottom() && bbox.height != 0.0 {
        sy = (cursor.y - bbox.min_y()) / bbox.height;
    }
    if keep_aspect && handle.is_corner() {
        // Lock aspect to the larger-magnitude scale.
        let s = if sx.abs() > sy.abs() { sx } else { sy };
        sx = s;
        sy = s;
    }
    (sx, sy)
}

/// Apply a scale to an element box `orig`, holding fixed the selection-box edge
/// *opposite* the dragged `handle`. Every selected element scales about the same
/// stationary anchor, so a multi-selection resizes as one rigid group.
fn apply_scale(orig: Rect, bbox: Rect, handle: Handle, (sx, sy): (f64, f64)) -> Rect {
    // Stationary anchor: the side that does NOT move under this handle.
    // Dragging the left edge keeps the right edge fixed, and vice versa; an edge
    // handle that doesn't touch an axis leaves that axis' min as the anchor
    // (its scale is 1.0 anyway, so the choice is immaterial).
    let anchor_x = if handle.affects_left() {
        bbox.max_x()
    } else {
        bbox.min_x()
    };
    let anchor_y = if handle.affects_top() {
        bbox.max_y()
    } else {
        bbox.min_y()
    };
    let nx = anchor_x + (orig.min_x() - anchor_x) * sx;
    let ny = anchor_y + (orig.min_y() - anchor_y) * sy;
    let nw = orig.width * sx;
    let nh = orig.height * sy;
    // Normalize negative sizes (flips) so width/height stay positive and the
    // origin remains the min corner.
    Rect::new(nx, ny, nw, nh)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::ElementKind;
    use std::f64::consts::PI;

    fn down(x: f64, y: f64) -> InputEvent {
        InputEvent::PointerDown {
            pos: Point::new(x, y),
            button: PointerButton::Primary,
            mods: Modifiers::default(),
        }
    }
    fn down_shift(x: f64, y: f64) -> InputEvent {
        InputEvent::PointerDown {
            pos: Point::new(x, y),
            button: PointerButton::Primary,
            mods: Modifiers {
                shift: true,
                ..Default::default()
            },
        }
    }
    fn mv(x: f64, y: f64) -> InputEvent {
        InputEvent::PointerMove {
            pos: Point::new(x, y),
            mods: Modifiers::default(),
        }
    }
    fn up(x: f64, y: f64) -> InputEvent {
        InputEvent::PointerUp {
            pos: Point::new(x, y),
            button: PointerButton::Primary,
            mods: Modifiers::default(),
        }
    }

    fn fresh() -> (InteractionState, Scene, Viewport) {
        (
            InteractionState::default(),
            Scene::new(),
            Viewport::default(),
        )
    }

    fn drive(
        st: &mut InteractionState,
        sc: &mut Scene,
        vp: &mut Viewport,
        tool: Tool,
        events: impl IntoIterator<Item = InputEvent>,
    ) -> Vec<InteractionResult> {
        events
            .into_iter()
            .map(|e| st.handle(sc, vp, tool, e))
            .collect()
    }

    fn add_rect(sc: &mut Scene, id: &str, x: f64, y: f64, w: f64, h: f64) {
        sc.insert(Element::new(
            ElementId::from(id),
            1,
            x,
            y,
            w,
            h,
            ElementKind::Rectangle,
        ));
    }

    // --- Creation -------------------------------------------------------

    #[test]
    fn drag_creates_rectangle() {
        let (mut st, mut sc, mut vp) = fresh();
        let results = drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Rectangle,
            [down(10.0, 10.0), mv(110.0, 60.0), up(110.0, 60.0)],
        );
        assert!(results[0].begins_undo);
        assert_eq!(sc.iter_live().count(), 1);
        let el = sc.iter_live().next().unwrap();
        assert_eq!(el.raw_box(), Rect::new(10.0, 10.0, 100.0, 50.0));
        // pointer-up reports a Created commit
        assert!(matches!(results[2].commit, Some(Commit::Created(_))));
        // it's selected
        assert_eq!(st.selection().len(), 1);
    }

    #[test]
    fn click_without_drag_creates_nothing() {
        let (mut st, mut sc, mut vp) = fresh();
        let results = drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Rectangle,
            [down(10.0, 10.0), up(10.0, 10.0)],
        );
        assert_eq!(sc.iter_live().count(), 0);
        assert!(results[1].commit.is_none());
    }

    #[test]
    fn drag_creates_arrow_with_points() {
        let (mut st, mut sc, mut vp) = fresh();
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Arrow,
            [down(0.0, 0.0), mv(50.0, 30.0), up(50.0, 30.0)],
        );
        let el = sc.iter_live().next().unwrap();
        if let ElementKind::Arrow(l) = &el.kind {
            assert_eq!(l.points, vec![Point::ORIGIN, Point::new(50.0, 30.0)]);
        } else {
            panic!("expected arrow");
        }
    }

    #[test]
    fn freedraw_accumulates_and_commits_even_without_move() {
        let (mut st, mut sc, mut vp) = fresh();
        // A bare tap with freedraw is a legit dot.
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Freedraw,
            [down(5.0, 5.0), up(5.0, 5.0)],
        );
        assert_eq!(sc.iter_live().count(), 1);
    }

    // --- Selection ------------------------------------------------------

    #[test]
    fn click_selects_topmost() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 0.0, 0.0, 100.0, 100.0);
        add_rect(&mut sc, "b", 20.0, 20.0, 50.0, 50.0); // on top, overlapping
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(30.0, 30.0), up(30.0, 30.0)],
        );
        assert_eq!(st.selection().len(), 1);
        assert!(st.selection().contains(&ElementId::from("b")));
    }

    #[test]
    fn shift_click_toggles() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 0.0, 0.0, 40.0, 40.0);
        add_rect(&mut sc, "b", 100.0, 0.0, 40.0, 40.0);
        // select a, then shift-select b, then shift-click b again to remove
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(10.0, 10.0), up(10.0, 10.0)],
        );
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down_shift(110.0, 10.0), up(110.0, 10.0)],
        );
        assert_eq!(st.selection().len(), 2);
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down_shift(110.0, 10.0), up(110.0, 10.0)],
        );
        assert_eq!(st.selection().len(), 1);
        assert!(st.selection().contains(&ElementId::from("a")));
    }

    #[test]
    fn click_empty_clears_selection() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 0.0, 0.0, 40.0, 40.0);
        st.set_selection([ElementId::from("a")]);
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(500.0, 500.0), up(500.0, 500.0)],
        );
        assert!(st.selection().is_empty());
    }

    #[test]
    fn marquee_selects_enclosed() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 10.0, 10.0, 20.0, 20.0); // fully inside
        add_rect(&mut sc, "b", 200.0, 200.0, 20.0, 20.0); // outside
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(0.0, 0.0), mv(100.0, 100.0), up(100.0, 100.0)],
        );
        assert_eq!(st.selection().len(), 1);
        assert!(st.selection().contains(&ElementId::from("a")));
    }

    // --- Move -----------------------------------------------------------

    #[test]
    fn drag_moves_selection() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 0.0, 0.0, 40.0, 40.0);
        let results = drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(20.0, 20.0), mv(70.0, 90.0), up(70.0, 90.0)],
        );
        let el = sc.get(&ElementId::from("a")).unwrap();
        assert_eq!(el.x, 50.0);
        assert_eq!(el.y, 70.0);
        assert!(matches!(results[2].commit, Some(Commit::Moved(_))));
    }

    #[test]
    fn moving_multiple_keeps_relative_offsets() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 0.0, 0.0, 40.0, 40.0);
        add_rect(&mut sc, "b", 100.0, 0.0, 40.0, 40.0);
        st.set_selection([ElementId::from("a"), ElementId::from("b")]);
        // click on a (already selected) and drag
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(10.0, 10.0), mv(10.0, 60.0), up(10.0, 60.0)],
        );
        assert_eq!(sc.get(&ElementId::from("a")).unwrap().y, 50.0);
        assert_eq!(sc.get(&ElementId::from("b")).unwrap().y, 50.0);
        assert_eq!(sc.get(&ElementId::from("b")).unwrap().x, 100.0);
    }

    // --- Resize ---------------------------------------------------------

    #[test]
    fn drag_se_handle_resizes() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 100.0, 100.0, 100.0, 100.0);
        st.set_selection([ElementId::from("a")]);
        // SE handle is at screen (200,200) under identity viewport.
        let results = drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(200.0, 200.0), mv(300.0, 250.0), up(300.0, 250.0)],
        );
        let el = sc.get(&ElementId::from("a")).unwrap();
        // width doubled, height 1.5x; top-left anchored.
        assert_eq!(el.x, 100.0);
        assert_eq!(el.y, 100.0);
        assert!((el.width - 200.0).abs() < 1e-9, "w={}", el.width);
        assert!((el.height - 150.0).abs() < 1e-9, "h={}", el.height);
        assert!(matches!(results[2].commit, Some(Commit::Resized(_))));
    }

    #[test]
    fn drag_nw_handle_anchors_opposite_corner() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 100.0, 100.0, 100.0, 100.0);
        st.set_selection([ElementId::from("a")]);
        // NW handle at (100,100); drag it to (150,150): box should shrink, with
        // the SE corner (200,200) fixed.
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(100.0, 100.0), mv(150.0, 150.0), up(150.0, 150.0)],
        );
        let el = sc.get(&ElementId::from("a")).unwrap();
        assert!((el.x - 150.0).abs() < 1e-9, "x={}", el.x);
        assert!((el.y - 150.0).abs() < 1e-9, "y={}", el.y);
        assert!((el.raw_box().max_x() - 200.0).abs() < 1e-9);
        assert!((el.raw_box().max_y() - 200.0).abs() < 1e-9);
    }

    // --- Rotate ---------------------------------------------------------

    #[test]
    fn drag_rotation_handle_rotates() {
        let (mut st, mut sc, mut vp) = fresh();
        add_rect(&mut sc, "a", 0.0, 0.0, 100.0, 100.0);
        st.set_selection([ElementId::from("a")]);
        let layout = st.handle_layout(&sc, &vp).unwrap();
        let rot = layout.center(Handle::Rotation);
        // Grab the rotation handle, then move 90deg clockwise around the pivot
        // (50,50). Start above center; move to the right of center.
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [
                down(rot.x, rot.y),
                mv(150.0, 50.0), // to the right of pivot => +90deg
                up(150.0, 50.0),
            ],
        );
        let el = sc.get(&ElementId::from("a")).unwrap();
        assert!((el.angle - PI / 2.0).abs() < 1e-6, "angle={}", el.angle);
    }

    // --- Pan / zoom -----------------------------------------------------

    #[test]
    fn pan_tool_scrolls_viewport() {
        let (mut st, mut sc, mut vp) = fresh();
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Pan,
            [down(100.0, 100.0), mv(60.0, 80.0), up(60.0, 80.0)],
        );
        // dragging content left/up by (40,20) scrolls scene right/down by same.
        assert_eq!(vp.scroll, Vec2::new(40.0, 20.0));
    }

    #[test]
    fn space_drag_pans_even_with_select_tool() {
        let (mut st, mut sc, mut vp) = fresh();
        st.handle(
            &mut sc,
            &mut vp,
            Tool::Select,
            InputEvent::KeyDown {
                key: Key::Char(' '),
                mods: Modifiers::default(),
            },
        );
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Select,
            [down(100.0, 100.0), mv(90.0, 100.0), up(90.0, 100.0)],
        );
        assert_eq!(vp.scroll, Vec2::new(10.0, 0.0));
    }

    #[test]
    fn middle_button_pans() {
        let (mut st, mut sc, mut vp) = fresh();
        let mid_down = InputEvent::PointerDown {
            pos: Point::new(100.0, 100.0),
            button: PointerButton::Middle,
            mods: Modifiers::default(),
        };
        st.handle(&mut sc, &mut vp, Tool::Select, mid_down);
        st.handle(&mut sc, &mut vp, Tool::Select, mv(80.0, 100.0));
        assert_eq!(vp.scroll, Vec2::new(20.0, 0.0));
    }

    #[test]
    fn ctrl_wheel_zooms_toward_cursor() {
        let (mut st, mut sc, mut vp) = fresh();
        let anchor = Point::new(200.0, 200.0);
        let before = vp.screen_to_scene(anchor);
        st.handle(
            &mut sc,
            &mut vp,
            Tool::Select,
            InputEvent::Wheel {
                delta: Vec2::new(0.0, -100.0),
                pos: anchor,
                mods: Modifiers {
                    ctrl: true,
                    ..Default::default()
                },
            },
        );
        assert!(vp.zoom > 1.0, "zoom={}", vp.zoom);
        let after = vp.screen_to_scene(anchor);
        assert!((before.x - after.x).abs() < 1e-6);
        assert!((before.y - after.y).abs() < 1e-6);
    }

    #[test]
    fn plain_wheel_pans() {
        let (mut st, mut sc, mut vp) = fresh();
        st.handle(
            &mut sc,
            &mut vp,
            Tool::Select,
            InputEvent::Wheel {
                delta: Vec2::new(30.0, 40.0),
                pos: Point::new(0.0, 0.0),
                mods: Modifiers::default(),
            },
        );
        assert_eq!(vp.scroll, Vec2::new(30.0, 40.0));
    }

    #[test]
    fn escape_cancels_gesture() {
        let (mut st, mut sc, mut vp) = fresh();
        st.handle(&mut sc, &mut vp, Tool::Rectangle, down(0.0, 0.0));
        assert!(st.is_interacting());
        let r = st.handle(
            &mut sc,
            &mut vp,
            Tool::Rectangle,
            InputEvent::KeyDown {
                key: Key::Escape,
                mods: Modifiers::default(),
            },
        );
        assert!(!st.is_interacting());
        assert!(r.view_changed);
    }

    #[test]
    fn ids_are_deterministic() {
        let (mut st, mut sc, mut vp) = fresh();
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Rectangle,
            [down(0.0, 0.0), mv(10.0, 10.0), up(10.0, 10.0)],
        );
        drive(
            &mut st,
            &mut sc,
            &mut vp,
            Tool::Rectangle,
            [down(20.0, 20.0), mv(30.0, 30.0), up(30.0, 30.0)],
        );
        let ids: Vec<_> = sc.order().iter().map(|i| i.as_str().to_string()).collect();
        assert_eq!(ids, ["el-0", "el-1"]);
    }
}
