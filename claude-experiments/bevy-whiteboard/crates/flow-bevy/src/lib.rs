//! Flow-powered whiteboard. Public lib so integration tests in `tests/` can
//! construct the app and exercise its plugins — see [`build_app`]. The
//! binary entrypoint is a one-liner in `main.rs`.

pub mod bridge;
pub mod camera;
pub mod canvas;
pub mod edges;
pub mod errors;
pub mod examples;
pub mod examples_menu;
pub mod gadgets;
pub mod hud;
pub mod inspector;
pub mod nodes;
pub mod palette;
pub mod probes;
pub mod theme;
pub mod timeline;
pub mod tool;
pub mod visual;

use std::path::PathBuf;

use bevy::prelude::*;

/// Construct the Bevy app with every plugin wired up. Does *not* call
/// `.run()` — the binary does that, tests call `.update()` themselves.
///
/// If `canvas` is supplied, the app boots from that `.whiteboard`
/// directory instead of the built-in demo example. When the path is
/// invalid or the canvas fails to load, the error is logged and the
/// app falls back to the empty canvas (no demo) so the user can see
/// the error surface and recover from the UI.
pub fn build_app(canvas: Option<PathBuf>) -> App {
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
    .add_plugins(FlowBevyPlugins);
    match canvas {
        Some(path) => app.add_plugins(CanvasSeedPlugin(path)),
        None => app.add_plugins(DemoSeedPlugin),
    };
    app
}

/// Seed the canvas on startup by firing the `LoadExample` message for
/// the default scenario. Tests deliberately skip this so they start
/// from an empty canvas. Intentionally lives in lib.rs (not nodes.rs)
/// because the seed now dispatches through the examples pipeline, not
/// directly into the node/edge systems.
pub struct DemoSeedPlugin;
impl Plugin for DemoSeedPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, seed_default);
    }
}

fn seed_default(mut writer: bevy::ecs::message::MessageWriter<examples::LoadExample>) {
    writer.write(examples::LoadExample(examples::Example::ThreeLaneFanout));
}

/// Seed the canvas from a `.whiteboard` directory. The path is
/// resolved + loaded at startup; a loader failure logs an error and
/// leaves the canvas empty (the user can then use the palette to
/// load a built-in example or retry via the file menu).
pub struct CanvasSeedPlugin(pub PathBuf);
impl Plugin for CanvasSeedPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PendingCanvas(Some(self.0.clone())))
            .add_systems(Startup, canvas::seed_from_path);
    }
}

#[derive(Resource)]
pub(crate) struct PendingCanvas(pub Option<PathBuf>);

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
            timeline::TimelinePlugin,
            probes::ProbesPlugin,
            errors::ErrorsPlugin,
            examples::ExamplesPlugin,
            examples_menu::ExamplesMenuPlugin,
        ));
    }
}
