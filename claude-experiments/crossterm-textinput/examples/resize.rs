use crossterm_textinput::TextInput;
use crossterm::style::Color;
use std::io::Result;

fn main() -> Result<()> {
    println!("Inline Text Input Demo");
    println!("This text input appears inline with your terminal output");
    println!("Use Option+Enter for new lines, Enter to submit, Esc to exit");
    println!();

    let mut text_input = TextInput::new(8)
        .with_border_color(Color::Green)
        .with_text_color(Color::Yellow);

    let result = text_input.run_inline()?;
    
    println!("You entered:");
    println!("{}", result);
    
    Ok(())
}