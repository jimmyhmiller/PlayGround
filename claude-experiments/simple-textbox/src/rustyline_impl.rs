use crossterm::{
    terminal::{self, ClearType},
    ExecutableCommand,
};
use rustyline::{config::Configurer, Editor, Result};
use std::io::stdout;

pub struct RustylineTextBox {
    editor: Editor<(), rustyline::history::DefaultHistory>,
}

impl RustylineTextBox {
    pub fn new() -> Result<Self> {
        let mut editor = Editor::new()?;
        
        // Enable multiline editing
        editor.set_auto_add_history(true);
        
        Ok(Self { editor })
    }
    
    pub fn run(&mut self) -> Result<String> {
        // Clear from current position down
        stdout().execute(terminal::Clear(ClearType::FromCursorDown)).unwrap();
        
        // Draw top border
        let (width, _) = terminal::size().unwrap();
        print!("╭");
        for _ in 1..width-1 {
            print!("─");
        }
        println!("╮");
        
        // Use rustyline with custom prompt
        let readline = self.editor.readline("│ > ");
        
        match readline {
            Ok(line) => {
                // Draw bottom border after input
                print!("╰");
                for _ in 1..width-1 {
                    print!("─");
                }
                println!("╯");
                
                Ok(line)
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                // Draw bottom border on Ctrl-C
                print!("╰");
                for _ in 1..width-1 {
                    print!("─");
                }
                println!("╯");
                
                Ok(String::new())
            }
            Err(err) => Err(err),
        }
    }
}