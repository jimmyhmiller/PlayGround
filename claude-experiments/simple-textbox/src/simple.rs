use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{self, ClearType},
    ExecutableCommand,
};
use std::io::{self, stdout, Write};

pub struct SimpleTextBox {
    buffer: String,
    cursor_pos: usize,
    start_row: u16,
    width: u16,
}

impl SimpleTextBox {
    pub fn new() -> io::Result<Self> {
        let (_, start_row) = crossterm::cursor::position()?;
        let (width, _) = terminal::size()?;
        
        Ok(Self {
            buffer: String::new(),
            cursor_pos: 0,
            start_row,
            width,
        })
    }
    
    pub fn run(&mut self) -> io::Result<String> {
        crossterm::terminal::enable_raw_mode()?;
        
        // Clear area and draw initial box
        stdout().execute(terminal::Clear(ClearType::FromCursorDown))?;
        self.draw_box()?;
        
        loop {
            // Update cursor position
            self.position_cursor()?;
            
            // Handle input
            if let Event::Key(key_event) = event::read()? {
                match (key_event.code, key_event.modifiers) {
                    (KeyCode::Enter, KeyModifiers::NONE) => {
                        crossterm::terminal::disable_raw_mode()?;
                        self.move_to_end()?;
                        return Ok(self.buffer.clone());
                    }
                    (KeyCode::Enter, KeyModifiers::ALT) => {
                        self.insert_char('\n');
                        self.draw_box()?;
                    }
                    (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                        crossterm::terminal::disable_raw_mode()?;
                        self.move_to_end()?;
                        return Ok(String::new());
                    }
                    (KeyCode::Char(c), _) => {
                        self.insert_char(c);
                        self.draw_box()?;
                    }
                    (KeyCode::Backspace, _) => {
                        if self.delete_char() {
                            self.draw_box()?;
                        }
                    }
                    (KeyCode::Left, _) => {
                        if self.cursor_pos > 0 {
                            self.cursor_pos -= 1;
                        }
                    }
                    (KeyCode::Right, _) => {
                        if self.cursor_pos < self.buffer.len() {
                            self.cursor_pos += 1;
                        }
                    }
                    (KeyCode::Up, _) => {
                        self.cursor_up();
                    }
                    (KeyCode::Down, _) => {
                        self.cursor_down();
                    }
                    (KeyCode::Esc, _) => {
                        crossterm::terminal::disable_raw_mode()?;
                        self.move_to_end()?;
                        return Ok(String::new());
                    }
                    _ => {}
                }
            }
        }
    }
    
    fn insert_char(&mut self, c: char) {
        self.buffer.insert(self.cursor_pos, c);
        self.cursor_pos += 1;
    }
    
    fn delete_char(&mut self) -> bool {
        if self.cursor_pos > 0 {
            self.cursor_pos -= 1;
            self.buffer.remove(self.cursor_pos);
            true
        } else {
            false
        }
    }
    
    fn cursor_up(&mut self) {
        let lines: Vec<&str> = self.buffer.split('\n').collect();
        let (current_line, current_col) = self.get_cursor_line_col();
        
        if current_line > 0 {
            let target_line = current_line - 1;
            let target_col = current_col.min(lines[target_line].len());
            self.cursor_pos = self.get_pos_from_line_col(target_line, target_col);
        }
    }
    
    fn cursor_down(&mut self) {
        let lines: Vec<&str> = self.buffer.split('\n').collect();
        let (current_line, current_col) = self.get_cursor_line_col();
        
        if current_line < lines.len() - 1 {
            let target_line = current_line + 1;
            let target_col = current_col.min(lines[target_line].len());
            self.cursor_pos = self.get_pos_from_line_col(target_line, target_col);
        }
    }
    
    fn get_cursor_line_col(&self) -> (usize, usize) {
        let mut line = 0;
        let mut col = 0;
        
        for (i, ch) in self.buffer.chars().enumerate() {
            if i == self.cursor_pos {
                break;
            }
            if ch == '\n' {
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
        }
        
        (line, col)
    }
    
    fn get_pos_from_line_col(&self, target_line: usize, target_col: usize) -> usize {
        let mut pos = 0;
        let mut line = 0;
        let mut col = 0;
        
        for ch in self.buffer.chars() {
            if line == target_line && col == target_col {
                break;
            }
            if ch == '\n' {
                if line == target_line {
                    break;
                }
                line += 1;
                col = 0;
            } else {
                col += 1;
            }
            pos += 1;
        }
        
        pos
    }
    
    fn draw_box(&mut self) -> io::Result<()> {
        // Update width in case of resize
        let (width, _) = terminal::size()?;
        self.width = width;
        
        // Move to start and clear down
        stdout().execute(crossterm::cursor::MoveTo(0, self.start_row))?;
        stdout().execute(terminal::Clear(ClearType::FromCursorDown))?;
        
        let lines: Vec<&str> = if self.buffer.is_empty() {
            vec![""]
        } else {
            self.buffer.split('\n').collect()
        };
        
        // Top border
        print!("╭");
        for _ in 1..self.width-1 {
            print!("─");
        }
        println!("╮");
        
        // Content lines
        for (i, line) in lines.iter().enumerate() {
            print!("│ ");
            if i == 0 {
                print!("> ");
            } else {
                print!("  ");
            }
            
            if i == 0 && self.buffer.is_empty() {
                print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                let content_len = 30;
                for _ in content_len..self.width as usize - 2 {
                    print!(" ");
                }
            } else {
                print!("{}", line);
                let content_len = line.len() + 4;
                for _ in content_len..self.width as usize - 2 {
                    print!(" ");
                }
            }
            println!(" │");
        }
        
        // Bottom border
        print!("╰");
        for _ in 1..self.width-1 {
            print!("─");
        }
        println!("╯");
        
        stdout().flush()?;
        Ok(())
    }
    
    fn position_cursor(&self) -> io::Result<()> {
        let (line, col) = self.get_cursor_line_col();
        let cursor_x = 4 + col as u16;
        let cursor_y = self.start_row + 1 + line as u16;
        stdout().execute(crossterm::cursor::MoveTo(cursor_x, cursor_y))?;
        Ok(())
    }
    
    fn move_to_end(&self) -> io::Result<()> {
        let lines = self.buffer.split('\n').count().max(1);
        let final_row = self.start_row + lines as u16 + 2;
        stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
        Ok(())
    }
}