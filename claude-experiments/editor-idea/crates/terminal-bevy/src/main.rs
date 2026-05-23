use bevy::prelude::*;
use terminal_bevy::TerminalPlugin;

fn main() {
    // Self-exec daemon mode: when the editor needs a per-session daemon
    // it re-execs this same binary with `--daemon <session_id> <cmd...>`.
    // Dispatch before touching Bevy so the daemon process never loads
    // the GUI stack.
    let mut args = std::env::args().skip(1);
    if args.next().as_deref() == Some("--daemon") {
        let session_id: u64 = args
            .next()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| {
                eprintln!(
                    "usage: terminal --daemon <session_id> <program> [args...]"
                );
                std::process::exit(2);
            });
        let command: Vec<String> = args.collect();
        if command.is_empty() {
            eprintln!("terminal --daemon: missing program to run");
            std::process::exit(2);
        }
        terminal_daemon::daemon::run(session_id, command);
    }

    eprintln!("[terminal-bevy] startup marker: bundle-identity-test");

    let mut app = App::new();
    // Register the per-project asset source for style-bevy BEFORE
    // DefaultPlugins, since AssetPlugin (part of DefaultPlugins)
    // freezes the source registry once it's added.
    if let Some(data_dir) = terminal_bevy::data_dir() {
        style_bevy::register_style_asset_source(&mut app, data_dir.join("projects"));
    }
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "terminal-bevy".into(),
            resolution: (1200u32, 760u32).into(),
            ..default()
        }),
        ..default()
    }));
    app.add_plugins(TerminalPlugin);
    // Subscribe to Claude Code hook events from the central bus. Any
    // system in this app (or its panes) can react by reading
    // MessageReader<claude_bus_bevy::ClaudeBusEvent>. If the bus isn't
    // running the subscriber thread just retries in the background —
    // nothing in the app blocks on it.
    app.add_plugins(claude_bus_bevy::BusEventPlugin::default());
    app.run();
}
