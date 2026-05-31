//! Bridge Bevy input events to flame-graph renderer state.
//!
//! Each entity carrying `FlameGraph` gets its own `FlameGraphInput`. The
//! plugin runs one system that, per entity:
//!  1. Reads the Bevy cursor position relative to a panel origin you supply
//!     (zero when full-window).
//!  2. Translates mouse motion → pan / hover, wheel → scroll/zoom, clicks →
//!     hit-tests, and keyboard shortcuts (1-5 for tabs, f to flip direction,
//!     m for merge mode, a/0/Home/Esc to fit-all, +/- to zoom).
//!
//! If you want the host app to filter or rewrite events (e.g. only forward
//! when a UI panel has focus), set `FlameGraphInput::enabled = false` and
//! drive the renderer yourself via `FlameGraph::renderer_mut(...)`.

use bevy::input::keyboard::{Key, KeyboardInput};
use bevy::input::mouse::{MouseButton, MouseButtonInput, MouseScrollUnit, MouseWheel};
use bevy::input::ButtonState;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use flame_render::MainTab;

use crate::FlameGraph;

/// Attach this to the same entity that carries [`FlameGraph`] to drive it
/// from Bevy's input events. Defaults to enabled, anchored at (0,0), and
/// sized to the panel.
#[derive(Component, Debug, Clone)]
pub struct FlameGraphInput {
    /// Master switch. When `false` the input system skips this entity.
    pub enabled: bool,
    /// Top-left corner of the flame-graph panel in physical-window pixels.
    /// For full-window mode leave at `Vec2::ZERO`. For an embedded panel set
    /// this to where the panel is drawn in the Bevy window.
    pub panel_origin: Vec2,
    /// Logical size of the panel in physical-window pixels. Must match the
    /// `size` field on [`FlameGraph`].
    pub panel_size: Vec2,
    /// Internal: last known cursor position in panel-local pixels.
    cursor: Vec2,
    /// Internal: drag origin while LMB is held.
    drag_origin: Option<Vec2>,
}

impl Default for FlameGraphInput {
    fn default() -> Self {
        Self {
            enabled: true,
            panel_origin: Vec2::ZERO,
            panel_size: Vec2::ZERO,
            cursor: Vec2::ZERO,
            drag_origin: None,
        }
    }
}

impl FlameGraphInput {
    /// True if the last cursor position was inside the panel rectangle.
    pub fn cursor_in_panel(&self) -> bool {
        self.cursor.x >= 0.0
            && self.cursor.y >= 0.0
            && self.cursor.x < self.panel_size.x
            && self.cursor.y < self.panel_size.y
    }
}

/// System: route Bevy input to each entity's [`FlameGraph`]. Add via
/// [`crate::FlameGraphPlugin`]; you should not normally call it yourself.
#[allow(clippy::too_many_arguments)]
pub fn forward_input(
    windows: Query<&Window, With<PrimaryWindow>>,
    mut q: Query<(&mut FlameGraph, &mut FlameGraphInput)>,
    mut wheel: MessageReader<MouseWheel>,
    mut clicks: MessageReader<MouseButtonInput>,
    mut keys: MessageReader<KeyboardInput>,
) {
    let Ok(window) = windows.single() else { return };
    let cursor_window = window.cursor_position();

    // We have to fan-out the same event streams across every panel. Snapshot
    // first to avoid `MessageReader` consuming them after the first iteration.
    let wheel: Vec<_> = wheel.read().cloned().collect();
    let clicks: Vec<_> = clicks.read().cloned().collect();
    let keys: Vec<_> = keys.read().cloned().collect();

    for (mut flame, mut input) in &mut q {
        if !input.enabled {
            continue;
        }

        // Cursor: translate to panel-local. Note: Bevy's cursor_position is in
        // logical pixels; we treat that as physical here, which matches the
        // common case of `scale_factor = 1` and is what the renderer expects.
        if let Some(cw) = cursor_window {
            let local = cw - input.panel_origin;
            input.cursor = local;
        }
        let cursor = input.cursor;

        // Hover / drag based on cursor motion. (Bevy reports it as a state
        // each frame rather than an event stream, so we apply unconditionally.)
        if input.cursor_in_panel() {
            if let Some(origin) = input.drag_origin {
                let dx = cursor.x - origin.x;
                let dy = cursor.y - origin.y;
                let r = flame.renderer_mut();
                r.viewport.pan_x_px(dx);
                r.viewport.pan_y_px(dy);
                r.clamp_viewport();
                r.rebuild_instances();
                input.drag_origin = Some(cursor);
                flame.mark_dirty();
            } else {
                let r = flame.renderer_mut();
                let hit = r.hit_test(cursor.x, cursor.y);
                let prev = r.hovered;
                r.set_hover(hit);
                if r.hovered != prev {
                    flame.mark_dirty();
                }
            }
        }

        // Wheel.
        for w in &wheel {
            if !input.cursor_in_panel() {
                continue;
            }
            let (dx, dy) = match w.unit {
                MouseScrollUnit::Line => (w.x * 30.0, w.y * 30.0),
                MouseScrollUnit::Pixel => (w.x, w.y),
            };
            let r = flame.renderer_mut();
            match r.active_tab {
                MainTab::CallTree => {
                    if dy != 0.0 {
                        r.pan_call_tree(dy);
                        r.rebuild_instances();
                    }
                }
                MainTab::Sequence => {
                    if dy != 0.0 {
                        r.pan_sequence(-dy);
                        r.rebuild_instances();
                    }
                }
                MainTab::Flame if r.cursor_in_inspector(cursor.x) => {
                    if dy != 0.0 {
                        r.pan_sidebar(-dy);
                        r.rebuild_instances();
                    }
                }
                _ => {
                    if dx != 0.0 {
                        r.viewport.pan_x_px(dx);
                    }
                    if dy != 0.0 {
                        r.viewport.pan_y_px(dy);
                    }
                    r.clamp_viewport();
                    r.rebuild_instances();
                }
            }
            flame.mark_dirty();
        }

        // Clicks. We intentionally don't model right-button-drag; only left
        // click + drag-pan, matching the standalone viewer.
        for c in &clicks {
            if c.button != MouseButton::Left {
                continue;
            }
            if !input.cursor_in_panel() {
                if matches!(c.state, ButtonState::Released) {
                    input.drag_origin = None;
                }
                continue;
            }
            match c.state {
                ButtonState::Pressed => {
                    handle_left_press(&mut flame, &mut input, cursor);
                }
                ButtonState::Released => {
                    input.drag_origin = None;
                }
            }
        }

        // Keyboard shortcuts (1-5, f, m, a/0/Home/Esc, +/-).
        for k in &keys {
            if !matches!(k.state, ButtonState::Pressed) {
                continue;
            }
            let r = flame.renderer_mut();
            let mut handled = true;
            match &k.logical_key {
                Key::Character(s) => match s.as_str() {
                    "1" | "2" | "3" | "4" | "5" => {
                        if let Ok(idx) = s.parse::<usize>() {
                            if let Some(&tab) = MainTab::ALL.get(idx.saturating_sub(1)) {
                                r.set_tab(tab);
                                r.rebuild_instances();
                            }
                        }
                    }
                    "f" | "F" => {
                        r.flip_direction();
                        r.rebuild_instances();
                    }
                    "m" | "M" => {
                        r.toggle_merge_mode();
                        r.rebuild_instances();
                    }
                    "a" | "A" | "0" => {
                        r.fit_all();
                        r.rebuild_instances();
                    }
                    "+" | "=" => {
                        r.viewport.zoom_at(r.viewport.size_px.0 * 0.5, 0.7);
                        r.clamp_viewport();
                        r.rebuild_instances();
                    }
                    "-" | "_" => {
                        r.viewport.zoom_at(r.viewport.size_px.0 * 0.5, 1.43);
                        r.clamp_viewport();
                        r.rebuild_instances();
                    }
                    _ => handled = false,
                },
                Key::Home | Key::Escape => {
                    r.fit_all();
                    r.rebuild_instances();
                }
                Key::ArrowLeft => {
                    let pan = r.viewport.size_px.0 * 0.10;
                    r.viewport.pan_x_px(pan);
                    r.clamp_viewport();
                    r.rebuild_instances();
                }
                Key::ArrowRight => {
                    let pan = r.viewport.size_px.0 * 0.10;
                    r.viewport.pan_x_px(-pan);
                    r.clamp_viewport();
                    r.rebuild_instances();
                }
                Key::ArrowUp => {
                    r.viewport.pan_y_px(20.0);
                    r.clamp_viewport();
                    r.rebuild_instances();
                }
                Key::ArrowDown => {
                    r.viewport.pan_y_px(-20.0);
                    r.clamp_viewport();
                    r.rebuild_instances();
                }
                _ => handled = false,
            }
            if handled {
                flame.mark_dirty();
            }
        }
    }
}

/// Mirror of the click-routing logic in `flame-viewer/src/main.rs`. Kept
/// inline here rather than pushed into `flame-render` because the priority
/// (top-tab → call-tree node → layout button → sidebar tab → group row →
/// track header → inspector → timeline) is a viewer-policy decision; the
/// renderer exposes only the individual hit-tests.
fn handle_left_press(flame: &mut FlameGraph, input: &mut FlameGraphInput, cursor: Vec2) {
    let r = flame.renderer_mut();
    if let Some(tab) = r.hit_test_inspector_tab(cursor.x, cursor.y) {
        r.set_tab(tab);
        r.rebuild_instances();
    } else if r.active_tab == MainTab::CallTree {
        if let Some(node_idx) = r.hit_test_call_tree(cursor.x, cursor.y) {
            r.toggle_tree_node(node_idx);
            r.rebuild_instances();
        }
    } else if let Some(mode) = r.hit_test_layout_button(cursor.x, cursor.y) {
        r.set_layout_mode(mode);
        r.rebuild_instances();
    } else if let Some(tab) = r.hit_test_sidebar_tab(cursor.x, cursor.y) {
        r.set_sidebar_tab(tab);
        r.rebuild_instances();
    } else if let Some(pick) = r.hit_test_group_row(cursor.x, cursor.y) {
        r.set_group_key(pick);
        r.rebuild_instances();
    } else if let Some(track_id) = r.hit_test_track_header(cursor.x, cursor.y) {
        r.toggle_track_collapsed(track_id);
        r.rebuild_instances();
    } else if r.cursor_in_inspector(cursor.x) {
        if let Some(slice_idx) = r.hit_test_inspector(cursor.x, cursor.y) {
            r.select_slice(Some(slice_idx));
            if let Some(p) = &r.profile {
                let s = p.slices.start_ns[slice_idx as usize];
                let d = p.slices.dur_ns[slice_idx as usize];
                let mid = s as f64 + d as f64 * 0.5;
                r.viewport.start_ns =
                    mid - r.viewport.size_px.0 as f64 * r.viewport.ns_per_pixel * 0.5;
            }
            r.clamp_viewport();
            r.rebuild_instances();
        }
    } else {
        let hit_inst = r.hit_test(cursor.x, cursor.y);
        let slice_idx = hit_inst.and_then(|i| r.instance_to_slice(i));
        r.select_slice(slice_idx);
        input.drag_origin = Some(cursor);
        r.rebuild_instances();
    }
    flame.mark_dirty();
}
