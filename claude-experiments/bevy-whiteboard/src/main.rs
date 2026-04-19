mod bridge;
mod camera;
mod chrome;
mod edges;
mod inspector;
mod nodes;
mod palette;
mod sim;
mod simulation;
mod theme;
mod tool;

use bevy::prelude::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Living Whiteboard".into(),
                resolution: (1400u32, 900u32).into(),
                // macOS: keep the traffic-light buttons but float them over a
                // transparent titlebar so the paper colour reaches the top
                // edge. `fullsize_content_view` is what extends our SVG
                // canvas under the bar; `titlebar_transparent` removes the
                // standard chrome backing; `titlebar_show_title` hides the
                // window title text (we have our own poster header).
                titlebar_transparent: true,
                fullsize_content_view: true,
                titlebar_show_title: false,
                ..default()
            }),
            ..default()
        }))
        // ClearColor is overwritten by `theme::sync_clear_color` whenever the
        // Theme resource changes; this initial value just avoids a one-frame
        // flash of unstyled background before the first sync runs.
        .insert_resource(ClearColor(Color::srgb(0.91, 0.87, 0.77)))
        .init_resource::<tool::ActiveTool>()
        .add_plugins(theme::ThemePlugin)
        .init_resource::<tool::ActiveColor>()
        .init_resource::<nodes::NodeRegistry>()
        .add_plugins((
            camera::CameraPlugin,
            palette::PalettePlugin,
            chrome::ChromePlugin,
            inspector::InspectorPlugin,
            nodes::NodesPlugin,
            edges::EdgesPlugin,
            simulation::SimulationPlugin,
        ))
        .run();
}
