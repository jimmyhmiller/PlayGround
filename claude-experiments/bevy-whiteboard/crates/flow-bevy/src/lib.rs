//! Flow-powered whiteboard. Public lib so integration tests in `tests/` can
//! construct the app and exercise its plugins — see [`build_app`]. The
//! binary entrypoint is a one-liner in `main.rs`.

pub mod bridge;
pub mod camera;
pub mod edges;
pub mod gadgets;
pub mod hud;
pub mod inspector;
pub mod nodes;
pub mod palette;
pub mod probes;
pub mod theme;
pub mod tool;

use bevy::prelude::*;

/// Construct the Bevy app with every plugin wired up. Does *not* call
/// `.run()` — the binary does that, tests call `.update()` themselves.
pub fn build_app() -> App {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "Flow Whiteboard".into(),
            resolution: (1400u32, 900u32).into(),
            titlebar_transparent: true,
            fullsize_content_view: true,
            titlebar_show_title: false,
            ..default()
        }),
        ..default()
    }))
    .add_plugins(FlowBevyPlugins)
    .add_plugins(nodes::DemoSeedPlugin);
    app
}

/// Plugin variant of [`build_app`] for integration tests that want to start
/// from a [`poster_ui::testing::test_app_headless`] and layer the flow-bevy
/// behavior on top. Includes everything [`build_app`] adds except
/// `DefaultPlugins` (the test harness supplies its own) and `PosterUiPlugin`.
pub struct FlowBevyPlugins;
impl Plugin for FlowBevyPlugins {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<ClearColor>() {
            app.insert_resource(ClearColor(Color::srgb(0.91, 0.87, 0.77)));
        }
        app.add_plugins(poster_ui::PosterUiPlugin).add_plugins((
            tool::ToolPlugin,
            camera::CameraPlugin,
            bridge::FlowBridgePlugin,
            nodes::NodesPlugin,
            edges::EdgesPlugin,
            palette::PalettePlugin,
            hud::HudPlugin,
            inspector::InspectorPlugin,
            probes::ProbesPlugin,
        ));
    }
}
