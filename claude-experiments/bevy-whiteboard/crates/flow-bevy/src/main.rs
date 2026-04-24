//! Binary shim — real app construction lives in lib.rs so tests share it.
//!
//! Optional CLI: `flow-bevy [PATH]` where PATH is a `.whiteboard`
//! directory. When supplied, that canvas's sim + visual state seed the
//! app instead of the built-in demo example.

use std::path::PathBuf;

fn main() {
    let canvas_path: Option<PathBuf> = std::env::args()
        .nth(1)
        .filter(|a| !a.starts_with('-'))
        .map(PathBuf::from);
    flow_bevy::build_app(canvas_path).run();
}
