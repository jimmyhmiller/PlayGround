use simple_textbox::ClaudeTextBoxSimple;
use std::io::{self, Write};

fn main() -> std::io::Result<()> {
    let result = ClaudeTextBoxSimple::run_efficient()?;
    
    // Don't clear or move - let the textbox stay where it is
    // Just position cursor after it and print the result for further processing
    if !result.is_empty() {
        print!("{}", result);
        io::stdout().flush()?;
    }
    
    Ok(())
}