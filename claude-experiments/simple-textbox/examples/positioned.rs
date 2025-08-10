use simple_textbox::ClaudeTextBoxSimple;
use crossterm::{terminal, ExecutableCommand};
use std::io::{self, stdout};

fn main() -> io::Result<()> {
    // Clear screen
    stdout().execute(terminal::Clear(terminal::ClearType::All))?;
    
    // Position cursor at row 5, column 10
    stdout().execute(crossterm::cursor::MoveTo(10, 5))?;
    
    // Run the textbox at a specific position
    let result = ClaudeTextBoxSimple::run_at_position(5, 10)?;
    
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