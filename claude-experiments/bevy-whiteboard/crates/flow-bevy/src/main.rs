// Kept as a thin shim — the real app construction lives in lib.rs as
// `flow_bevy::build_app` so integration tests can share it.
fn main() {
    flow_bevy::build_app().run();
}
