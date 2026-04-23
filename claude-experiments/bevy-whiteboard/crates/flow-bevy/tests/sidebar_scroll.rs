//! Regression test for the right sidebar's scroll. The body holds
//! tool buttons + data palette + inspector + footer; on smaller
//! windows the content overflows. `ScrollPane` + Bevy's
//! `ScrollPosition` should scroll it, but a classic flexbox gotcha
//! (`min-size: auto` on a flex child) used to prevent any overflow
//! from happening, so the wheel went nowhere.
//!
//! This test: position cursor over the palette body, emit a wheel
//! message, assert `ScrollPosition.y` went positive. Then emit a
//! negative wheel, assert it came back toward 0. If the flex fix
//! regresses, this flips red.

mod common;

use bevy::input::ButtonState;
use bevy::input::mouse::{MouseScrollUnit, MouseWheel};
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use common::make_app;
use poster_ui::{PanelBodyBg, ScrollPane};

/// Project a UI entity's center into logical (window-space) px for
/// the primary window, so we can stamp the cursor onto it.
fn center_of(app: &mut App, entity: Entity) -> Vec2 {
    let world = app.world_mut();
    let (xf, _cn) = world
        .query::<(&bevy::ui::UiGlobalTransform, &bevy::ui::ComputedNode)>()
        .get(world, entity)
        .expect("entity missing UiGlobalTransform / ComputedNode");
    let translation_physical = xf.translation;
    let window_entity = world
        .query_filtered::<Entity, With<PrimaryWindow>>()
        .iter(world)
        .next()
        .expect("no PrimaryWindow");
    let scale = world.get::<Window>(window_entity).unwrap().scale_factor();
    Vec2::new(translation_physical.x / scale, translation_physical.y / scale)
}

fn set_cursor(app: &mut App, logical: Vec2) -> Entity {
    let window_entity = {
        let world = app.world_mut();
        world
            .query_filtered::<Entity, With<PrimaryWindow>>()
            .iter(world)
            .next()
            .expect("no PrimaryWindow")
    };
    {
        let mut win = app
            .world_mut()
            .get_mut::<Window>(window_entity)
            .expect("window missing");
        win.set_cursor_position(Some(logical));
    }
    app.world_mut().write_message(bevy::window::CursorMoved {
        window: window_entity,
        position: logical,
        delta: None,
    });
    app.update();
    window_entity
}

fn send_wheel(app: &mut App, window: Entity, dy: f32) {
    app.world_mut().write_message(MouseWheel {
        unit: MouseScrollUnit::Line,
        x: 0.0,
        y: dy,
        window,
    });
    // Suppress unused-import warning on ButtonState — it's here so
    // that if the test grows to emit button events, the import is
    // ready to go without re-adding.
    let _ = ButtonState::Pressed;
    app.update();
}

fn scroll_pane_entity_and_y(app: &mut App) -> (Entity, f32) {
    let world = app.world_mut();
    let mut q = world
        .query_filtered::<(Entity, &ScrollPosition), (With<ScrollPane>, With<PanelBodyBg>)>();
    let (e, pos) = q.iter(world).next().expect("no ScrollPane / PanelBodyBg entity");
    (e, pos.y)
}

#[test]
fn sidebar_scroll_wheel_moves_scroll_position() {
    let mut app = make_app();

    // Place cursor over the right sidebar's scrollable body.
    let (pane_entity, y_before) = scroll_pane_entity_and_y(&mut app);
    assert_eq!(y_before, 0.0, "scroll position should start at 0");

    let center = center_of(&mut app, pane_entity);
    let window = set_cursor(&mut app, center);

    // Scroll down: positive wheel delta increases scroll y.
    send_wheel(&mut app, window, -5.0);
    let (_, y_after_down) = scroll_pane_entity_and_y(&mut app);
    assert!(
        y_after_down > 0.0,
        "wheel-down should produce positive scroll offset, got {}",
        y_after_down
    );

    // Scroll up: negative-direction wheel should reduce it, clamped at 0.
    send_wheel(&mut app, window, 10.0);
    let (_, y_after_up) = scroll_pane_entity_and_y(&mut app);
    assert!(
        y_after_up < y_after_down,
        "wheel-up should reduce scroll offset: before={} after={}",
        y_after_down, y_after_up,
    );
    assert!(
        y_after_up >= 0.0,
        "scroll offset should clamp at 0, got {}",
        y_after_up
    );
}

/// Counter-example: wheel events outside the sidebar's bounds must
/// NOT affect the sidebar's ScrollPosition. Ensures the cursor-in-
/// bounds routing in `scroll_panes_on_wheel` is working, not just
/// "scroll always applies to the first pane." Without this, a fix
/// that made scrolling globally-on-any-wheel would pass the first
/// test and silently break other scroll panes' isolation.
#[test]
fn sidebar_scroll_ignores_wheel_outside_bounds() {
    let mut app = make_app();

    // Cursor at origin in logical coords — which maps to some point
    // inside the canvas region (center of window), NOT over the
    // right sidebar.
    let window = set_cursor(&mut app, Vec2::new(10.0, 10.0));

    send_wheel(&mut app, window, -5.0);

    let (_, y) = scroll_pane_entity_and_y(&mut app);
    assert_eq!(
        y, 0.0,
        "wheel outside the sidebar must not scroll it, got y={}",
        y
    );
}
