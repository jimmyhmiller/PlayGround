//! Test harness for Bevy apps that use `poster-ui` (or any Bevy 0.18 UI, really).
//!
//! Enabled via the `testing` feature. Gives you a headless [`App`] with the
//! full `DefaultPlugins` (so `UiPlugin`, `TextPlugin`, layout, and
//! `Interaction` all work) but without a real window or GPU backend — tests
//! run fast and don't need a display.
//!
//! Two click flavours are supported:
//!
//! 1. **UI-button clicks** — write `Interaction::Pressed` directly onto the
//!    target entity. Systems that watch `Changed<Interaction>` react on the
//!    next `app.update()`. See [`click_by_marker`].
//!
//! 2. **Canvas clicks** — project a world-space position through the
//!    camera to a screen coordinate, set the window's cursor there, and
//!    emit `MouseButtonInput` events. The Bevy input plugin turns those
//!    into `ButtonInput<MouseButton>::just_pressed` in `PreUpdate`,
//!    exactly as a real click would. See [`simulate_canvas_click`].
//!
//! Typical shape:
//! ```ignore
//! let mut app = test_app_headless();
//! app.add_plugins(my_app::AppPlugins);
//! app.update();                                          // run Startup
//!
//! click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Drop(Kind::Generator));
//! simulate_canvas_click(&mut app, Vec2::new(0.0, 0.0));
//!
//! let sim = &app.world().resource::<FlowSim>().sim;
//! assert_eq!(sim.nodes.len(), 1);
//! ```

use bevy::input::ButtonState;
use bevy::input::mouse::MouseButtonInput;
use bevy::prelude::*;
use bevy::render::RenderPlugin;
use bevy::render::settings::{RenderCreation, WgpuSettings};
use bevy::window::PrimaryWindow;
use bevy::winit::WinitPlugin;

/// Build a headless Bevy app suitable for integration tests. `DefaultPlugins`
/// is included (so UI layout and text measurement work), but the winit event
/// loop and GPU backend are disabled. The primary window is 1400×900 so
/// `ComputedNode` layouts resolve to sensible sizes.
///
/// Pass the result through `.add_plugins(...)` with your own plugins, then
/// call `app.update()` once to run `Startup`.
pub fn test_app_headless() -> App {
    let mut app = App::new();
    app.add_plugins(
        DefaultPlugins
            .build()
            .disable::<WinitPlugin>()
            .set(RenderPlugin {
                render_creation: RenderCreation::Automatic(WgpuSettings {
                    backends: None,
                    ..default()
                }),
                ..default()
            })
            .set(WindowPlugin {
                primary_window: Some(Window {
                    resolution: (1400u32, 900u32).into(),
                    ..default()
                }),
                ..default()
            }),
    );
    app
}

/// Click the first UI entity whose marker component matches `predicate`.
///
/// Mirrors a real click: looks up the entity's layout centre, moves the
/// window cursor there, and emits mouse press + release events. Bevy's
/// `ui_focus_system` (PreUpdate) sets `Interaction::Pressed` based on cursor
/// position × pressed button, then the consumer's `Changed<Interaction>`
/// handler reacts in Update.
///
/// Returns `true` if a matching entity was found and clicked.
///
/// ```ignore
/// click_by_marker::<ToolBtn, _>(&mut app, |m| m.0 == Tool::Connect);
/// ```
pub fn click_by_marker<M, F>(app: &mut App, predicate: F) -> bool
where
    M: Component,
    F: Fn(&M) -> bool,
{
    let target = {
        let world = app.world_mut();
        let mut q = world.query::<(Entity, &M)>();
        q.iter(world)
            .find(|(_, m)| predicate(m))
            .map(|(e, _)| e)
    };
    let Some(entity) = target else { return false };

    // Pull the UI element's centre out of its transform. `UiGlobalTransform`
    // is in physical pixels — divide by the window's scale factor to convert
    // to the logical coordinates `Window::cursor_position` reports.
    let world = app.world_mut();
    let (centre_physical, _size) = {
        let mut q = world.query::<(&bevy::ui::UiGlobalTransform, &bevy::ui::ComputedNode)>();
        let (xf, cn) = q
            .get(world, entity)
            .expect("target entity missing UiGlobalTransform / ComputedNode — did you click_by_marker on a non-UI entity?");
        (xf.translation, cn.size)
    };

    let window_entity = {
        let mut q = world.query_filtered::<Entity, With<PrimaryWindow>>();
        q.iter(world)
            .next()
            .expect("click_by_marker: no PrimaryWindow")
    };
    let scale = world
        .get::<Window>(window_entity)
        .expect("window entity has no Window")
        .scale_factor();
    let cursor_logical = Vec2::new(centre_physical.x / scale, centre_physical.y / scale);

    {
        let mut win = world
            .get_mut::<Window>(window_entity)
            .expect("window missing");
        win.set_cursor_position(Some(cursor_logical));
    }

    // Emit a CursorMoved event so the picking system updates its cached
    // cursor before the button press is evaluated.
    world.write_message(bevy::window::CursorMoved {
        window: window_entity,
        position: cursor_logical,
        delta: None,
    });
    app.update();

    // Press → Interaction becomes Pressed (in PreUpdate). Handler fires
    // in Update.
    app.world_mut().write_message(MouseButtonInput {
        button: MouseButton::Left,
        state: ButtonState::Pressed,
        window: window_entity,
    });
    app.update();

    // Release so the button doesn't stay "held" between calls.
    app.world_mut().write_message(MouseButtonInput {
        button: MouseButton::Left,
        state: ButtonState::Released,
        window: window_entity,
    });
    app.update();

    true
}

/// Reset every `Interaction` component in the world to `None`. Useful
/// between test steps to clear sticky hover/press state that could trigger
/// stray `pointer_over_ui()` gates on the next click.
pub fn reset_interactions(app: &mut App) {
    let world = app.world_mut();
    let entities: Vec<Entity> = {
        let mut q = world.query_filtered::<Entity, With<Interaction>>();
        q.iter(world).collect()
    };
    for e in entities {
        world.entity_mut(e).insert(Interaction::None);
    }
}

/// Simulate a left-click at a world-space position. Projects through the
/// primary camera to find a viewport coordinate, moves the window cursor
/// there, and emits `MouseButtonInput` press + release events across two
/// `app.update()` calls.
///
/// Panics in three cases, all of which used to silently swallow the click:
///  * No camera / no primary window exists.
///  * The world position projects outside the window's viewport.
///  * The projected screen position lands under a UI element (any node with
///    an `Interaction` whose bounding box contains the point) — the app's
///    own `pointer_over_ui` gate would discard the click.
pub fn simulate_canvas_click(app: &mut App, world_pos: Vec2) {
    // 1. Project the world position into screen space using the primary camera.
    let screen = {
        let world = app.world_mut();
        let mut q = world.query::<(&Camera, &GlobalTransform)>();
        let (cam, tf) = q
            .iter(world)
            .next()
            .expect("simulate_canvas_click: no Camera found in test app");
        cam.world_to_viewport(tf, world_pos.extend(0.0))
            .expect("simulate_canvas_click: world_to_viewport failed")
    };

    // 2. Find the primary window entity and stamp the cursor position onto
    //    its `Window` component. `cursor_position()` will read this back.
    let window_entity = {
        let world = app.world_mut();
        let mut q = world.query_filtered::<Entity, With<PrimaryWindow>>();
        q.iter(world)
            .next()
            .expect("simulate_canvas_click: no PrimaryWindow")
    };
    let (win_w, win_h, scale) = {
        let win = app
            .world()
            .get::<Window>(window_entity)
            .expect("window entity has no Window component");
        (win.width(), win.height(), win.scale_factor())
    };

    // 2a. Reject world positions whose projection falls off-viewport.
    //     Without this the click "lands" at coordinates no camera can invert,
    //     and the app's own `cursor_to_world` helper silently returns None.
    if screen.x < 0.0 || screen.y < 0.0 || screen.x >= win_w || screen.y >= win_h {
        panic!(
            "simulate_canvas_click: world ({:.0}, {:.0}) projects to screen \
             ({:.0}, {:.0}), outside the {}×{} window. Move the target point \
             inside the viewport (world x ∈ [{:.0}, {:.0}], y ∈ [{:.0}, {:.0}] \
             for a centred Camera2d at default scale).",
            world_pos.x, world_pos.y, screen.x, screen.y, win_w, win_h,
            -win_w * 0.5, win_w * 0.5, -win_h * 0.5, win_h * 0.5,
        );
    }

    // 2b. Reject projections that fall under a UI element (any entity with
    //     an `Interaction` component whose layout bbox contains the point).
    //     The app's `pointer_over_ui` gate would filter this click out, so
    //     failing loudly beats silently doing nothing.
    let cursor_physical = screen * scale;
    let hit = {
        let world = app.world_mut();
        let mut q = world.query_filtered::<
            (&bevy::ui::ComputedNode, &bevy::ui::UiGlobalTransform),
            With<Interaction>,
        >();
        q.iter(world)
            .find(|(cn, xf)| {
                let half = cn.size * 0.5;
                let center = xf.translation;
                cursor_physical.x >= center.x - half.x
                    && cursor_physical.x <= center.x + half.x
                    && cursor_physical.y >= center.y - half.y
                    && cursor_physical.y <= center.y + half.y
            })
            .map(|(cn, xf)| (cn.size, xf.translation))
    };
    if let Some((size, center)) = hit {
        panic!(
            "simulate_canvas_click: world ({:.0}, {:.0}) projects to screen \
             ({:.0}, {:.0}), which lies under a UI element centred at \
             ({:.0}, {:.0}) sized {:.0}×{:.0} (physical px). The app's \
             `pointer_over_ui` guard will block the click. Move the target \
             to a world position whose viewport projection is clear of all UI.",
            world_pos.x, world_pos.y, screen.x, screen.y,
            center.x, center.y, size.x, size.y,
        );
    }

    {
        let world = app.world_mut();
        let mut win = world
            .get_mut::<Window>(window_entity)
            .expect("window entity has no Window component");
        win.set_cursor_position(Some(screen));
    }

    // 3. Emit CursorMoved so bevy's UI focus / picking system updates its
    //    cached pointer position — otherwise `Interaction` on the last
    //    UI button we clicked can linger as Hovered and make
    //    `pointer_over_ui()` gates fire on what should be a canvas click.
    app.world_mut().write_message(bevy::window::CursorMoved {
        window: window_entity,
        position: screen,
        delta: None,
    });

    // 4. Press. Writing the event lets Bevy's PreUpdate input system
    //    populate `ButtonInput<MouseButton>::just_pressed` the same way a
    //    real mouse click would.
    let press = MouseButtonInput {
        button: MouseButton::Left,
        state: ButtonState::Pressed,
        window: window_entity,
    };
    app.world_mut().write_message(press);
    app.update();

    // 4. Release on the next frame so nothing stays "held" between tests.
    let release = MouseButtonInput {
        button: MouseButton::Left,
        state: ButtonState::Released,
        window: window_entity,
    };
    app.world_mut().write_message(release);
    app.update();
}

/// Convenience: run `app.update()` `n` times. Useful when you want to let a
/// few frames of simulation / layout elapse before asserting.
pub fn tick(app: &mut App, n: usize) {
    for _ in 0..n {
        app.update();
    }
}
