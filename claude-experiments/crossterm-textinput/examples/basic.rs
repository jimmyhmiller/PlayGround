use crossterm_textinput::TextInput;
use crossterm::style::Color;
use std::io::Result;

fn main() -> Result<()> {
    println!("Basic Text Input Demo");
    println!("Use Option+Enter for new lines, Enter to submit, Esc to exit");
    println!();

    let mut text_input = TextInput::new(5)
        .with_border_color(Color::Cyan)
        .with_text_color(Color::White);

    let result = text_input.run_inline()?;
    
    println!("You entered:");
    println!("{}", result);
    
    Ok(())
}