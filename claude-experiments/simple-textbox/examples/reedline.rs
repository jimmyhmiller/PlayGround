use simple_textbox::ClaudeTextBox;
use std::io;

fn main() -> io::Result<()> {
    let mut textbox = ClaudeTextBox::new()?;
    let result = textbox.run()?;
    
    if !result.is_empty() {
        println!("You entered:\n{}", result);
    } else {
        println!("Cancelled");
    }
    
    Ok(())
}