//! Contract tests for `simulate_canvas_click`'s failure modes. If these
//! stop panicking, a future test will silently drop the click instead of
//! loudly failing — please keep them green.

mod common;

use bevy::prelude::*;
use common::make_app;
use poster_ui::testing::simulate_canvas_click;

#[test]
#[should_panic(expected = "outside the 1400")]
fn off_viewport_click_panics() {
    let mut app = make_app();
    // World (10000, 0) projects far past the right edge of the viewport.
    simulate_canvas_click(&mut app, Vec2::new(10000.0, 0.0));
}

#[test]
#[should_panic(expected = "under a UI element")]
fn click_under_palette_panics() {
    let mut app = make_app();
    // World x=500 in a 1400-px window projects to screen x=1200, which sits
    // inside the 264-px-wide right-side palette (inset 20 px).
    simulate_canvas_click(&mut app, Vec2::new(500.0, 0.0));
}
