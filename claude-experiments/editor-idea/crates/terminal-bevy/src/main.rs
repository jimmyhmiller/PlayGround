use bevy::prelude::*;
use terminal_bevy::TerminalPlugin;

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
    app.run();
}
