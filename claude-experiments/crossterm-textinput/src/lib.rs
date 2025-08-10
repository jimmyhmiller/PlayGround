use crossterm::{
    cursor::{position, MoveTo},
    event::{Event, KeyCode, KeyEvent, KeyModifiers, poll, read},
    execute,
    style::{Color, Print, ResetColor, SetForegroundColor},
    terminal::{disable_raw_mode, enable_raw_mode, size, ScrollUp},
};
use std::io::{stdout, Result, Write};
use std::time::Duration;

#[derive(Debug, Clone)]
pub struct TextInput {
    content: Vec<String>,
    cursor_row: usize,
    cursor_col: usize,
    x: u16,
    y: u16,
    width: u16,
    height: u16,
    border_color: Color,
    text_color: Color,
    // Track what we actually drew so we can clear it exactly
    last_drawn_x: u16,
    last_drawn_y: u16,
    last_drawn_width: u16,
    last_drawn_height: u16,
}

impl TextInput {
    pub fn new(height: u16) -> Self {
        let (width, _) = size().unwrap_or((80, 24));
        Self {
            content: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
            x: 0,
            y: 0,
            width,
            height,
            border_color: Color::White,
            text_color: Color::White,
            last_drawn_x: 0,
            last_drawn_y: 0,
            last_drawn_width: 0,
            last_drawn_height: 0,
        }
    }

    pub fn new_with_width(width: u16, height: u16) -> Self {
        Self {
            content: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
            x: 0,
            y: 0,
            width,
            height,
            border_color: Color::White,
            text_color: Color::White,
            last_drawn_x: 0,
            last_drawn_y: 0,
            last_drawn_width: 0,
            last_drawn_height: 0,
        }
    }

    pub fn new_at_position(x: u16, y: u16, width: u16, height: u16) -> Self {
        Self {
            content: vec![String::new()],
            cursor_row: 0,
            cursor_col: 0,
            x,
            y,
            width,
            height,
            border_color: Color::White,
            text_color: Color::White,
            last_drawn_x: 0,
            last_drawn_y: 0,
            last_drawn_width: 0,
            last_drawn_height: 0,
        }
    }

    pub fn with_border_color(mut self, color: Color) -> Self {
        self.border_color = color;
        self
    }

    pub fn with_text_color(mut self, color: Color) -> Self {
        self.text_color = color;
        self
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }

    pub fn get_content(&self) -> String {
        self.content.join("\n")
    }

    pub fn clear(&mut self) {
        self.content = vec![String::new()];
        self.cursor_row = 0;
        self.cursor_col = 0;
    }

    fn draw_border(&self) -> Result<()> {
        let mut stdout = stdout();
        
        execute!(stdout, SetForegroundColor(self.border_color))?;
        
        execute!(stdout, MoveTo(self.x, self.y), Print("┌"))?;
        for _ in 1..self.width - 1 {
            execute!(stdout, Print("─"))?;
        }
        execute!(stdout, Print("┐"))?;
        
        for row in 1..self.height - 1 {
            execute!(stdout, MoveTo(self.x, self.y + row), Print("│"))?;
            execute!(stdout, MoveTo(self.x + self.width - 1, self.y + row), Print("│"))?;
        }
        
        execute!(stdout, MoveTo(self.x, self.y + self.height - 1), Print("└"))?;
        for _ in 1..self.width - 1 {
            execute!(stdout, Print("─"))?;
        }
        execute!(stdout, Print("┘"))?;
        
        execute!(stdout, ResetColor)?;
        Ok(())
    }

    fn draw_content(&self) -> Result<()> {
        let mut stdout = stdout();
        execute!(stdout, SetForegroundColor(self.text_color))?;
        
        let content_height = (self.height - 2) as usize;
        let content_width = (self.width - 2) as usize;
        
        for row in 0..content_height {
            execute!(stdout, MoveTo(self.x + 1, self.y + 1 + row as u16))?;
            
            if row < self.content.len() {
                let line = &self.content[row];
                let display_text = if line.len() > content_width {
                    &line[..content_width]
                } else {
                    line
                };
                execute!(stdout, Print(display_text))?;
                
                let remaining_chars = content_width - display_text.len();
                for _ in 0..remaining_chars {
                    execute!(stdout, Print(" "))?;
                }
            } else {
                for _ in 0..content_width {
                    execute!(stdout, Print(" "))?;
                }
            }
        }
        
        execute!(stdout, ResetColor)?;
        Ok(())
    }

    fn update_cursor(&self) -> Result<()> {
        let content_width = (self.width - 2) as usize;
        let display_col = if self.cursor_col > content_width {
            content_width
        } else {
            self.cursor_col
        };
        
        execute!(
            stdout(),
            MoveTo(
                self.x + 1 + display_col as u16,
                self.y + 1 + self.cursor_row as u16
            )
        )?;
        Ok(())
    }

    pub fn draw(&mut self) -> Result<()> {
        // Record what we're about to draw so we can clear it later
        self.last_drawn_x = self.x;
        self.last_drawn_y = self.y;
        self.last_drawn_width = self.width;
        self.last_drawn_height = self.height;
        
        self.draw_border()?;
        self.draw_content()?;
        self.update_cursor()?;
        Ok(())
    }

    fn clear_last_drawn(&self) -> Result<()> {
        // Clear the EXACT area where we previously drew, character by character
        if self.last_drawn_width > 0 && self.last_drawn_height > 0 {
            for row in 0..self.last_drawn_height {
                execute!(stdout(), MoveTo(self.last_drawn_x, self.last_drawn_y + row))?;
                for _ in 0..self.last_drawn_width {
                    execute!(stdout(), Print(" "))?;
                }
            }
            stdout().flush()?;
        }
        Ok(())
    }

    fn update_size(&mut self) -> Result<bool> {
        if let Ok((width, _)) = size() {
            if width != self.width {
                // Clear exactly what we drew before
                self.clear_last_drawn()?;
                // Update to new size
                self.width = width;
                // Draw at new size
                self.draw()?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    pub fn handle_input(&mut self) -> Result<bool> {
        match read()? {
            Event::Key(KeyEvent { code, modifiers, .. }) => {
                match code {
                    KeyCode::Char(c) => {
                        if modifiers.contains(KeyModifiers::ALT) && c == '\r' {
                            self.insert_newline();
                        } else if c == '\r' || c == '\n' {
                            return Ok(true);
                        } else {
                            self.insert_char(c);
                        }
                    }
                    KeyCode::Enter => {
                        if modifiers.contains(KeyModifiers::ALT) {
                            self.insert_newline();
                        } else {
                            return Ok(true);
                        }
                    }
                    KeyCode::Backspace => {
                        self.handle_backspace();
                    }
                    KeyCode::Delete => {
                        self.handle_delete();
                    }
                    KeyCode::Left => {
                        self.move_cursor_left();
                    }
                    KeyCode::Right => {
                        self.move_cursor_right();
                    }
                    KeyCode::Up => {
                        self.move_cursor_up();
                    }
                    KeyCode::Down => {
                        self.move_cursor_down();
                    }
                    KeyCode::Home => {
                        self.cursor_col = 0;
                    }
                    KeyCode::End => {
                        if self.cursor_row < self.content.len() {
                            self.cursor_col = self.content[self.cursor_row].len();
                        }
                    }
                    KeyCode::Esc => {
                        return Ok(true);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        Ok(false)
    }

    fn insert_char(&mut self, c: char) {
        if self.cursor_row < self.content.len() {
            let line = &mut self.content[self.cursor_row];
            if self.cursor_col <= line.len() {
                line.insert(self.cursor_col, c);
                self.cursor_col += 1;
            }
        }
    }

    fn insert_newline(&mut self) {
        if self.cursor_row < self.content.len() {
            let line = self.content[self.cursor_row].clone();
            let (left, right) = line.split_at(self.cursor_col);
            
            self.content[self.cursor_row] = left.to_string();
            self.content.insert(self.cursor_row + 1, right.to_string());
            
            self.cursor_row += 1;
            self.cursor_col = 0;
        } else {
            self.content.push(String::new());
            self.cursor_row = self.content.len() - 1;
            self.cursor_col = 0;
        }
    }

    fn handle_backspace(&mut self) {
        if self.cursor_col > 0 {
            if self.cursor_row < self.content.len() {
                self.content[self.cursor_row].remove(self.cursor_col - 1);
                self.cursor_col -= 1;
            }
        } else if self.cursor_row > 0 {
            let current_line = self.content.remove(self.cursor_row);
            self.cursor_row -= 1;
            self.cursor_col = self.content[self.cursor_row].len();
            self.content[self.cursor_row].push_str(&current_line);
        }
    }

    fn handle_delete(&mut self) {
        if self.cursor_row < self.content.len() {
            let line = &mut self.content[self.cursor_row];
            if self.cursor_col < line.len() {
                line.remove(self.cursor_col);
            } else if self.cursor_row + 1 < self.content.len() {
                let next_line = self.content.remove(self.cursor_row + 1);
                self.content[self.cursor_row].push_str(&next_line);
            }
        }
    }

    fn move_cursor_left(&mut self) {
        if self.cursor_col > 0 {
            self.cursor_col -= 1;
        } else if self.cursor_row > 0 {
            self.cursor_row -= 1;
            if self.cursor_row < self.content.len() {
                self.cursor_col = self.content[self.cursor_row].len();
            }
        }
    }

    fn move_cursor_right(&mut self) {
        if self.cursor_row < self.content.len() {
            let line = &self.content[self.cursor_row];
            if self.cursor_col < line.len() {
                self.cursor_col += 1;
            } else if self.cursor_row + 1 < self.content.len() {
                self.cursor_row += 1;
                self.cursor_col = 0;
            }
        }
    }

    fn move_cursor_up(&mut self) {
        if self.cursor_row > 0 {
            self.cursor_row -= 1;
            if self.cursor_row < self.content.len() {
                let line_len = self.content[self.cursor_row].len();
                if self.cursor_col > line_len {
                    self.cursor_col = line_len;
                }
            }
        }
    }

    fn move_cursor_down(&mut self) {
        if self.cursor_row + 1 < self.content.len() {
            self.cursor_row += 1;
            let line_len = self.content[self.cursor_row].len();
            if self.cursor_col > line_len {
                self.cursor_col = line_len;
            }
        }
    }


    fn ensure_space(&mut self) -> Result<()> {
        if let Ok((_, terminal_height)) = size() {
            if self.y + self.height > terminal_height {
                let lines_to_scroll = (self.y + self.height) - terminal_height;
                for _ in 0..lines_to_scroll {
                    execute!(stdout(), ScrollUp(1))?;
                }
                self.y = terminal_height - self.height;
            }
        }
        Ok(())
    }

    pub fn run(&mut self) -> Result<String> {
        if let Ok((_, y)) = position() {
            self.x = 0;
            self.y = y;
        }

        self.ensure_space()?;
        self.draw()?;

        enable_raw_mode()?;

        let mut resize_counter = 0;

        loop {
            if poll(Duration::from_millis(50))? {
                if self.handle_input()? {
                    break;
                }
                self.draw_content()?;
            } else {
                // Check resize periodically
                resize_counter += 1;
                if resize_counter >= 20 {  // Check every ~1 second
                    resize_counter = 0;
                    self.update_size()?;
                }
            }
        }

        disable_raw_mode()?;
        
        execute!(stdout(), MoveTo(0, self.y + self.height))?;
        println!();
        
        Ok(self.get_content())
    }

    pub fn run_inline(&mut self) -> Result<String> {
        if let Ok((_, y)) = position() {
            self.x = 0;
            self.y = y;
        }

        self.ensure_space()?;
        self.draw()?;

        enable_raw_mode()?;

        let mut resize_counter = 0;

        loop {
            if poll(Duration::from_millis(50))? {
                if self.handle_input()? {
                    break;
                }
                self.draw_content()?;
            } else {
                // Check resize periodically  
                resize_counter += 1;
                if resize_counter >= 20 {  // Check every ~1 second
                    resize_counter = 0;
                    self.update_size()?;
                }
            }
        }

        disable_raw_mode()?;
        
        execute!(stdout(), MoveTo(0, self.y + self.height))?;
        println!();
        
        Ok(self.get_content())
    }
}
