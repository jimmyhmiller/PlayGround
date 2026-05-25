//! Headless Bevy probe — same plugin we install in the real app, but
//! with a one-line MessageReader logger so we can see whether events
//! reach the Bevy side at all.

use bevy::prelude::*;

fn main() {
    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(claude_bus_bevy::BusEventPlugin::default());
    app.add_systems(Update, log_events);
    eprintln!("[probe] running — Ctrl-C to stop");
    app.run();
}

fn log_events(mut ev: MessageReader<claude_bus_bevy::ClaudeBusEvent>) {
    for e in ev.read() {
        eprintln!("[probe] seq={} kind={} sess={:?}", e.seq, e.kind, e.terminal_session_id);
    }
}
