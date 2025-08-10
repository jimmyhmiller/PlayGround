use simple_textbox::rustyline_impl::RustylineTextBox;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut textbox = RustylineTextBox::new()?;
    let result = textbox.run()?;
    
    if !result.is_empty() {
        println!("You entered:\n{}", result);
    } else {
        println!("Cancelled");
    }
    
    Ok(())
}