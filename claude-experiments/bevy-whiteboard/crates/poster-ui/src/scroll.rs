//! Vertical scroll pane primitive. Tag any UI node with [`ScrollPane`] — plus
//! `Overflow::scroll_y()` and a default `ScrollPosition` — and mousewheel
//! events inside that node's bounds will scroll it.
//!
//! We route by cursor-in-bounds rather than the `Interaction` component.
//! `Interaction` only flags the topmost hit, so a button or row *inside* the
//! pane would leave the pane's own interaction at `None` and swallow the
//! scroll. Bounding-box against the cursor is consistent regardless of what
//! child is under the pointer.

use bevy::prelude::*;

pub struct ScrollPlugin;

impl Plugin for ScrollPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, scroll_panes_on_wheel);
    }
}

/// Marker: route mousewheel events to this node when the cursor is over it.
/// Pair with `Overflow::scroll_y()` and `ScrollPosition::default()` on the
/// same entity, plus `Interaction::None` if you want the node to receive hit
/// testing.
#[derive(Component)]
pub struct ScrollPane;

fn scroll_panes_on_wheel(
    mut wheel: MessageReader<bevy::input::mouse::MouseWheel>,
    windows: Query<&Window>,
    mut panes: Query<
        (
            &bevy::ui::ComputedNode,
            &bevy::ui::UiGlobalTransform,
            &mut ScrollPosition,
        ),
        With<ScrollPane>,
    >,
) {
    let mut dy = 0.0f32;
    for ev in wheel.read() {
        // Normalise lines vs pixels to a consistent logical-pixel step.
        // ~20 logical px per wheel-line feels right on both trackpads and mice.
        let scale = match ev.unit {
            bevy::input::mouse::MouseScrollUnit::Line => 20.0,
            bevy::input::mouse::MouseScrollUnit::Pixel => 1.0,
        };
        dy -= ev.y * scale;
    }
    if dy.abs() < 1e-3 {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Some(cursor_logical) = win.cursor_position() else { return };
    let scale_f = win.scale_factor();
    // ComputedNode / UiGlobalTransform report physical pixels; the window
    // cursor is logical. Convert to a common space before bounding-box
    // testing.
    let cursor_px = cursor_logical * scale_f;
    for (computed, xform, mut pos) in panes.iter_mut() {
        let half = computed.size * 0.5;
        let center = xform.translation;
        let min = center - half;
        let max = center + half;
        if cursor_px.x >= min.x
            && cursor_px.x <= max.x
            && cursor_px.y >= min.y
            && cursor_px.y <= max.y
        {
            pos.y = (pos.y + dy).max(0.0);
            return;
        }
    }
}
