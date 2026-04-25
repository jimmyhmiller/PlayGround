use bevy::prelude::*;
use terminal_bevy::{
    setup_camera_and_font, spawn_terminal, FocusedTerminal, TerminalPlugin, TerminalRect,
};

fn main() {
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "terminal-bevy".into(),
            resolution: (1200u32, 760u32).into(),
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(TerminalPlugin);

    // Spawn two terminals on the canvas so the multi-instance / drag /
    // resize / focus paths get exercised right away.
    app.add_systems(
        Startup,
        spawn_initial_terminals.after(setup_camera_and_font),
    );
    app.run();
}

fn spawn_initial_terminals(world: &mut World) {
    let a = spawn_terminal(
        world,
        TerminalRect {
            pos: Vec2::new(40.0, 40.0),
            size: Vec2::new(640.0, 400.0),
            z: 1.0,
        },
    );
    let _b = spawn_terminal(
        world,
        TerminalRect {
            pos: Vec2::new(180.0, 200.0),
            size: Vec2::new(640.0, 400.0),
            z: 2.0,
        },
    );
    world.insert_resource(FocusedTerminal(Some(a)));
}
