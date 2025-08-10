use crossterm::{
    cursor::{MoveTo, position},
    event::{self, Event, KeyCode, KeyEventKind, poll},
    execute,
    style::Print,
    terminal::{size, enable_raw_mode, disable_raw_mode},
};
use std::io::{stdout, Result, Write};
use std::time::Duration;

struct InlineTerminalBox {
    start_x: u16,
    start_y: u16,
    width: u16,
    height: u16,
    last_drawn_width: u16,
}

impl InlineTerminalBox {
    fn new() -> Result<Self> {
        let (width, _) = size()?;
        let (start_x, start_y) = position()?;
        
        Ok(Self {
            start_x,
            start_y,
            width,
            height: 3,
            last_drawn_width: 0,
        })
    }
    
    fn clear_previous_draw(&self) -> Result<()> {
        if self.last_drawn_width > 0 {
            let mut stdout = stdout();
            for row in 0..self.height {
                execute!(stdout, MoveTo(self.start_x, self.start_y + row))?;
                execute!(stdout, Print(" ".repeat(self.last_drawn_width as usize)))?;
            }
            stdout.flush()?;
        }
        Ok(())
    }
    
    fn draw_box(&mut self) -> Result<()> {
        let mut stdout = stdout();
        
        self.last_drawn_width = self.width;
        
        // Top border
        execute!(stdout, MoveTo(self.start_x, self.start_y))?;
        execute!(stdout, Print("┌"))?;
        for _ in 1..self.width.saturating_sub(1) {
            execute!(stdout, Print("─"))?;
        }
        if self.width > 1 {
            execute!(stdout, Print("┐"))?;
        }
        
        // Middle line
        execute!(stdout, MoveTo(self.start_x, self.start_y + 1))?;
        execute!(stdout, Print("│"))?;
        for _ in 1..self.width.saturating_sub(1) {
            execute!(stdout, Print(" "))?;
        }
        if self.width > 1 {
            execute!(stdout, Print("│"))?;
        }
        
        // Bottom border
        execute!(stdout, MoveTo(self.start_x, self.start_y + 2))?;
        execute!(stdout, Print("└"))?;
        for _ in 1..self.width.saturating_sub(1) {
            execute!(stdout, Print("─"))?;
        }
        if self.width > 1 {
            execute!(stdout, Print("┘"))?;
        }
        
        stdout.flush()?;
        Ok(())
    }
    
    fn handle_resize(&mut self, new_width: u16) -> Result<()> {
        if new_width != self.width {
            self.clear_previous_draw()?;
            self.width = new_width;
            self.draw_box()?;
        }
        Ok(())
    }
    
    fn run(&mut self) -> Result<()> {
        enable_raw_mode()?;
        
        self.draw_box()?;
        
        loop {
            if poll(Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key) => {
                        if key.kind == KeyEventKind::Press {
                            match key.code {
                                KeyCode::Char('q') | KeyCode::Esc => break,
                                _ => {}
                            }
                        }
                    }
                    Event::Resize(new_width, _) => {
                        self.handle_resize(new_width)?;
                    }
                    _ => {}
                }
            }
        }
        
        disable_raw_mode()?;
        
        // Position cursor after the box
        execute!(stdout(), MoveTo(0, self.start_y + self.height))?;
        println!();
        
        Ok(())
    }
}

fn main() {
    println!("Starting inline terminal box demo...");
    println!("Resize terminal to test. Press 'q' to quit.");
    println!();
    
    if let Err(e) = InlineTerminalBox::new().and_then(|mut b| b.run()) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
    
    println!("Demo finished!");
}
