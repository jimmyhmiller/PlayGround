use minime::{
    editor::Editor,
    renderer::{full::CrosstermRenderer, Renderer},
    Result,
};
use simple_textbox::claude_styles::{ClaudeHeader, ClaudeMarginWithPlaceholder, ClaudeFooter};
use simple_textbox::custom_keybindings::ClaudeKeybinding;
use crossterm::event::{poll, read, Event};
use std::time::Duration;

fn main() -> Result<()> {
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();

    let mut renderer = CrosstermRenderer::render_to(&mut lock)
        .header(ClaudeHeader)
        .margin(ClaudeMarginWithPlaceholder)
        .footer(ClaudeFooter);

    let mut editor = Editor::default();
    
    // Custom read loop that handles resize events properly
    loop {
        renderer.draw(&editor)?;
        renderer.flush()?;

        // Check for events
        if poll(Duration::from_millis(50))? {
            let event = read()?;
            match event {
                Event::Resize(_, _) => {
                    // Resize detected, redraw on next iteration
                    continue;
                }
                Event::Key(key_event) => {
                    // Process the key event manually
                    if !ClaudeKeybinding::process_key_event(&mut editor, key_event)? {
                        break;
                    }
                }
                _ => {
                    // Other events, just continue
                    continue;
                }
            }
        }
    }
    
    // Move cursor to after the textbox without clearing
    println!();
    
    // Print the contents for further processing if needed
    if !editor.contents().is_empty() {
        print!("{}", editor.contents());
    }
    
    Ok(())
}