#![allow(clippy::single_match)]
mod editor;
mod event;
mod fps_counter;
// mod strokes;
mod handle_event;
mod keyboard;
mod native;
mod wasm_messenger;
mod widget;
mod window;
mod widget2;
mod color;
mod util;

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("This is only supported on macos")
}

fn main() {
    use editor::Editor;

    let mut editor = Editor::new();
    editor.setup();

    window::setup_window(editor);
}

// Thoughts on what to do
// Focus on panes and moving things around.
// Panes should be able to draw anything on them
// Allowing drawing on the background of the current pane
// Allow nested panes ala muse
// Allow drawing lots of different things
// Consider tables and graphs before messing with text editing
// Maybe even start with a code viewer?
// Be sure to capture all actions as data
// Be sure to have a reaction system
// Consider how undo and redo work with such an application
// Consider ink and arrows
// Consider rendering markdown
