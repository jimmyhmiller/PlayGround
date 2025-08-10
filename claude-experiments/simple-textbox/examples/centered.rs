use simple_textbox::ClaudeTextBoxSimple;
use crossterm::{terminal, ExecutableCommand};
use std::io::{self, stdout};

fn main() -> io::Result<()> {
    // Clear screen
    stdout().execute(terminal::Clear(terminal::ClearType::All))?;
    
    // Run the textbox centered on screen
    let result = ClaudeTextBoxSimple::run_centered()?;
    
    // Clear screen after input
    stdout().execute(terminal::Clear(terminal::ClearType::All))?;
    stdout().execute(crossterm::cursor::MoveTo(0, 0))?;
    
    if !result.is_empty() {
        println!("You entered:\n{}", result);
    } else {
        println!("Cancelled");
    }
    
    Ok(())
}