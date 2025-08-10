use simple_textbox::simple::SimpleTextBox;
use std::io;

fn main() -> io::Result<()> {
    let mut textbox = SimpleTextBox::new()?;
    let result = textbox.run()?;
    
    if !result.is_empty() {
        println!("You entered:\n{}", result);
    } else {
        println!("Cancelled");
    }
    
    Ok(())
}