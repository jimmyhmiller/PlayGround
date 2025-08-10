use simple_textbox::ClaudeTextBoxSimple;
use std::io;

fn main() -> io::Result<()> {
    let result = ClaudeTextBoxSimple::run()?;
    
    // Clear screen after input
    print!("\x1B[2J\x1B[H");
    
    if !result.is_empty() {
        println!("You entered:\n{}", result);
    } else {
        println!("Cancelled");
    }
    
    Ok(())
}