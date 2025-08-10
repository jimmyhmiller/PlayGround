use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    terminal::{self, ClearType},
    ExecutableCommand,
};

pub mod simple;
pub mod rustyline_impl;
pub mod claude_styles;
pub mod custom_keybindings;
use reedline::{
    default_emacs_keybindings, Color, EditCommand, Emacs, KeyCode as ReedlineKeyCode,
    KeyModifiers as ReedlineKeyModifiers, Keybindings, Prompt, PromptEditMode,
    PromptHistorySearch, Reedline, ReedlineEvent, Signal, ValidationResult, Validator,
};
use std::borrow::Cow;
use std::io::{self, stdout, Write};

pub struct ClaudeTextBox {
    reedline: Reedline,
    prompt: ClaudePrompt,
}

impl ClaudeTextBox {
    pub fn new() -> io::Result<Self> {
        let keybindings = configure_keybindings();
        let edit_mode = Box::new(Emacs::new(keybindings));
        
        let reedline = Reedline::create()
            .with_edit_mode(edit_mode)
            .with_validator(Box::new(ClaudeValidator))
            .with_quick_completions(false)
            .with_partial_completions(false)
            .with_ansi_colors(true)
            .with_highlighter(Box::new(ClaudeHighlighter));

        Ok(Self {
            reedline,
            prompt: ClaudePrompt::new(),
        })
    }

    pub fn run(&mut self) -> io::Result<String> {
        // Clear from current position down
        stdout().execute(terminal::Clear(ClearType::FromCursorDown))?;
        
        loop {
            match self.reedline.read_line(&self.prompt)? {
                Signal::Success(buffer) => {
                    // Draw bottom border after input
                    let (width, _) = terminal::size()?;
                    print!("╰");
                    for _ in 1..width-1 {
                        print!("─");
                    }
                    println!("╯");
                    return Ok(buffer);
                }
                Signal::CtrlD | Signal::CtrlC => {
                    // Draw bottom border after cancel
                    let (width, _) = terminal::size()?;
                    print!("╰");
                    for _ in 1..width-1 {
                        print!("─");
                    }
                    println!("╯");
                    return Ok(String::new());
                }
            }
        }
    }
}

fn configure_keybindings() -> Keybindings {
    let mut keybindings = default_emacs_keybindings();
    
    // Add Option+Enter (Alt+Enter) for new line
    keybindings.add_binding(
        ReedlineKeyModifiers::ALT,
        ReedlineKeyCode::Enter,
        ReedlineEvent::Edit(vec![EditCommand::InsertNewline]),
    );
    
    keybindings
}

pub struct ClaudePrompt {
    placeholder: String,
}

impl ClaudePrompt {
    pub fn new() -> Self {
        Self {
            placeholder: "Try \"fix typecheck errors\"".to_string(),
        }
    }
}

impl Prompt for ClaudePrompt {
    fn render_prompt_left(&self) -> Cow<str> {
        let (width, _) = terminal::size().unwrap_or((80, 24));
        
        let mut output = String::new();
        
        // Top border
        output.push('╭');
        for _ in 1..width-1 {
            output.push('─');
        }
        output.push_str("╮\n");
        
        // Start of middle line
        output.push_str("│ ");
        
        Cow::Owned(output)
    }

    fn render_prompt_right(&self) -> Cow<str> {
        Cow::Borrowed(" │")
    }

    fn render_prompt_indicator(&self, _prompt_mode: PromptEditMode) -> Cow<str> {
        Cow::Borrowed("> ")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<str> {
        Cow::Borrowed("│   ")
    }

    fn render_prompt_history_search_indicator(&self, _history_search: PromptHistorySearch) -> Cow<str> {
        Cow::Borrowed("search: ")
    }
    
    fn get_prompt_color(&self) -> Color {
        Color::DarkGrey
    }
    
    fn get_prompt_multiline_color(&self) -> nu_ansi_term::Color {
        nu_ansi_term::Color::DarkGray
    }
    
    fn get_indicator_color(&self) -> Color {
        Color::DarkGrey
    }
    
    fn get_prompt_right_color(&self) -> Color {
        Color::DarkGrey
    }

    fn right_prompt_on_last_line(&self) -> bool {
        true
    }
}

struct ClaudeValidator;

impl Validator for ClaudeValidator {
    fn validate(&self, _line: &str) -> ValidationResult {
        ValidationResult::Complete
    }
}

struct ClaudeHighlighter;

impl reedline::Highlighter for ClaudeHighlighter {
    fn highlight(&self, line: &str, _cursor: usize) -> reedline::StyledText {
        let mut styled = reedline::StyledText::new();
        styled.push((nu_ansi_term::Style::default(), line.to_string()));
        styled
    }
}

pub struct ClaudeTextBoxSimple;

impl ClaudeTextBoxSimple {
    pub fn run() -> io::Result<String> {
        Self::run_efficient()
    }
    
    pub fn run_efficient() -> io::Result<String> {
        crossterm::terminal::enable_raw_mode()?;
        
        let mut buffer = String::new();
        let mut cursor_pos = 0;
        let mut cursor_line = 0;
        let mut cursor_col = 0;
        
        let mut last_width = 0;
        let mut last_height = 0;
        let mut last_buffer_lines = 0;
        
        // Get starting position
        let (_, start_row) = crossterm::cursor::position()?;
        
        // Initial draw
        let (width, height) = terminal::size()?;
        last_width = width;
        last_height = height;
        
        stdout().execute(terminal::Clear(ClearType::FromCursorDown))?;
        Self::draw_initial_box(width, &buffer, start_row)?;
        last_buffer_lines = 1;
        
        loop {
            let (width, height) = terminal::size()?;
            
            // Always redraw everything to avoid artifacts - simpler and more reliable
            let current_lines = buffer.split('\n').count().max(1);
            let needs_redraw = width != last_width || height != last_height || current_lines != last_buffer_lines;
            
            if needs_redraw {
                // Clear from cursor down and redraw the entire box
                stdout().execute(crossterm::cursor::MoveTo(0, start_row))?;
                stdout().execute(terminal::Clear(ClearType::FromCursorDown))?;
                Self::draw_initial_box(width, &buffer, start_row)?;
                last_width = width;
                last_height = height;
                last_buffer_lines = current_lines;
            }
            
            // Calculate cursor position
            cursor_line = 0;
            cursor_col = 0;
            
            for ch in buffer.chars().take(cursor_pos) {
                if ch == '\n' {
                    cursor_line += 1;
                    cursor_col = 0;
                } else {
                    cursor_col += 1;
                }
            }
            
            // Position cursor
            let cursor_x = 4 + cursor_col as u16;
            let cursor_y = start_row + 1 + cursor_line as u16;
            stdout().execute(crossterm::cursor::MoveTo(cursor_x, cursor_y))?;
            
            // Handle input
            if event::poll(std::time::Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key_event) => {
                        match (key_event.code, key_event.modifiers) {
                            (KeyCode::Enter, KeyModifiers::NONE) => {
                                crossterm::terminal::disable_raw_mode()?;
                                let final_lines = buffer.split('\n').count().max(1);
                                let final_row = start_row + final_lines as u16 + 2;
                                stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
                                return Ok(buffer);
                            }
                            (KeyCode::Enter, KeyModifiers::ALT) => {
                                buffer.insert(cursor_pos, '\n');
                                cursor_pos += 1;
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                crossterm::terminal::disable_raw_mode()?;
                                let final_lines = buffer.split('\n').count().max(1);
                                let final_row = start_row + final_lines as u16 + 2;
                                stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
                                return Ok(String::new());
                            }
                            (KeyCode::Char(c), _) => {
                                buffer.insert(cursor_pos, c);
                                cursor_pos += 1;
                            }
                            (KeyCode::Backspace, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                    buffer.remove(cursor_pos);
                                }
                            }
                            (KeyCode::Left, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                }
                            }
                            (KeyCode::Right, _) => {
                                if cursor_pos < buffer.len() {
                                    cursor_pos += 1;
                                }
                            }
                            (KeyCode::Up, _) => {
                                // Move cursor up a line
                                if cursor_line > 0 {
                                    let lines: Vec<&str> = buffer.split('\n').collect();
                                    let target_line = cursor_line - 1;
                                    let target_col = cursor_col.min(lines[target_line].len());
                                    
                                    cursor_pos = lines.iter()
                                        .take(target_line)
                                        .map(|line| line.len() + 1)
                                        .sum::<usize>() + target_col;
                                }
                            }
                            (KeyCode::Down, _) => {
                                // Move cursor down a line
                                let lines: Vec<&str> = buffer.split('\n').collect();
                                if cursor_line < lines.len() - 1 {
                                    let target_line = cursor_line + 1;
                                    let target_col = cursor_col.min(lines[target_line].len());
                                    
                                    cursor_pos = lines.iter()
                                        .take(target_line)
                                        .map(|line| line.len() + 1)
                                        .sum::<usize>() + target_col;
                                }
                            }
                            (KeyCode::Esc, _) => {
                                crossterm::terminal::disable_raw_mode()?;
                                let final_lines = buffer.split('\n').count().max(1);
                                let final_row = start_row + final_lines as u16 + 2;
                                stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
                                return Ok(String::new());
                            }
                            _ => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        // Force redraw on next iteration by not updating variables yet
                        continue;
                    }
                    _ => {}
                }
            }
        }
    }
    
    fn draw_initial_box(width: u16, buffer: &str, start_row: u16) -> io::Result<()> {
        let lines: Vec<&str> = if buffer.is_empty() {
            vec![""]
        } else {
            buffer.split('\n').collect()
        };
        
        // Top border
        stdout().execute(crossterm::cursor::MoveTo(0, start_row))?;
        print!("╭");
        for _ in 1..width-1 {
            print!("─");
        }
        println!("╮");
        
        // Content lines
        for (i, line) in lines.iter().enumerate() {
            stdout().execute(crossterm::cursor::MoveTo(0, start_row + 1 + i as u16))?;
            if i == 0 {
                print!("│ > ");
            } else {
                print!("│   ");
            }
            
            if i == 0 && buffer.is_empty() {
                print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                let content_len = 30;
                for _ in content_len..width as usize - 2 {
                    print!(" ");
                }
            } else {
                print!("{}", line);
                let content_len = line.len() + 4;
                for _ in content_len..width as usize - 2 {
                    print!(" ");
                }
            }
            println!(" │");
        }
        
        // Bottom border
        let bottom_row = start_row + lines.len() as u16 + 1;
        stdout().execute(crossterm::cursor::MoveTo(0, bottom_row))?;
        print!("╰");
        for _ in 1..width-1 {
            print!("─");
        }
        println!("╯");
        
        stdout().flush()?;
        Ok(())
    }
    
    fn update_content_area(width: u16, buffer: &str, start_row: u16, old_lines: usize, new_lines: usize) -> io::Result<()> {
        let lines: Vec<&str> = if buffer.is_empty() {
            vec![""]
        } else {
            buffer.split('\n').collect()
        };
        
        // Clear old content area
        for i in 1..=old_lines.max(new_lines) + 1 {
            stdout().execute(crossterm::cursor::MoveTo(0, start_row + i as u16))?;
            stdout().execute(terminal::Clear(ClearType::CurrentLine))?;
        }
        
        // Draw new content lines
        for (i, line) in lines.iter().enumerate() {
            stdout().execute(crossterm::cursor::MoveTo(0, start_row + 1 + i as u16))?;
            if i == 0 {
                print!("│ > ");
            } else {
                print!("│   ");
            }
            
            if i == 0 && buffer.is_empty() {
                print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                let content_len = 30;
                for _ in content_len..width as usize - 2 {
                    print!(" ");
                }
            } else {
                print!("{}", line);
                let content_len = line.len() + 4;
                for _ in content_len..width as usize - 2 {
                    print!(" ");
                }
            }
            println!(" │");
        }
        
        // Redraw bottom border
        let bottom_row = start_row + lines.len() as u16 + 1;
        stdout().execute(crossterm::cursor::MoveTo(0, bottom_row))?;
        print!("╰");
        for _ in 1..width-1 {
            print!("─");
        }
        println!("╯");
        
        stdout().flush()?;
        Ok(())
    }
    
    fn update_current_line(width: u16, buffer: &str, start_row: u16, cursor_line: usize) -> io::Result<()> {
        let lines: Vec<&str> = if buffer.is_empty() {
            vec![""]
        } else {
            buffer.split('\n').collect()
        };
        
        if cursor_line < lines.len() {
            let line = lines[cursor_line];
            let line_row = start_row + 1 + cursor_line as u16;
            
            stdout().execute(crossterm::cursor::MoveTo(0, line_row))?;
            stdout().execute(terminal::Clear(ClearType::CurrentLine))?;
            
            if cursor_line == 0 {
                print!("│ > ");
            } else {
                print!("│   ");
            }
            
            if cursor_line == 0 && buffer.is_empty() {
                print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                let content_len = 30;
                for _ in content_len..width as usize - 2 {
                    print!(" ");
                }
            } else {
                print!("{}", line);
                let content_len = line.len() + 4;
                for _ in content_len..width as usize - 2 {
                    print!(" ");
                }
            }
            print!(" │");
            
            stdout().flush()?;
        }
        
        Ok(())
    }
    
    pub fn run_at_bottom() -> io::Result<String> {
        crossterm::terminal::enable_raw_mode()?;
        
        let mut buffer = String::new();
        let mut cursor_pos = 0;
        let mut cursor_line = 0;
        let mut cursor_col = 0;
        
        let mut needs_redraw = true;
        let mut last_width = 0;
        let mut last_height = 0;
        
        // Get current cursor position to start from where command was run
        let (start_col, start_row) = crossterm::cursor::position()?;
        
        loop {
            let (width, height) = terminal::size()?;
            
            // Check if terminal size changed
            if width != last_width || height != last_height {
                needs_redraw = true;
                last_width = width;
                last_height = height;
            }
            
            if needs_redraw {
                // Don't clear entire screen, just clear from current position down
                stdout().execute(crossterm::cursor::MoveTo(0, start_row))?;
                stdout().execute(terminal::Clear(ClearType::FromCursorDown))?;
                
                // Calculate box dimensions
                let lines: Vec<&str> = if buffer.is_empty() {
                    vec![""]
                } else {
                    buffer.split('\n').collect()
                };
                let box_height = lines.len().max(1) + 2; // +2 for borders
                
                // Draw top border
                stdout().execute(crossterm::cursor::MoveTo(0, start_row))?;
                print!("╭");
                for _ in 1..width-1 {
                    print!("─");
                }
                println!("╮");
                
                // Draw content lines
                for (i, line) in lines.iter().enumerate() {
                    stdout().execute(crossterm::cursor::MoveTo(0, start_row + 1 + i as u16))?;
                    if i == 0 {
                        print!("│ > ");
                    } else {
                        print!("│   ");
                    }
                    
                    // Show placeholder on first line if empty
                    if i == 0 && buffer.is_empty() {
                        print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                    } else {
                        print!("{}", line);
                    }
                    
                    // Fill the rest of the line
                    let content_len = if i == 0 && buffer.is_empty() { 
                        30 // length of placeholder + prompt
                    } else { 
                        line.len() + 4
                    };
                    let available_width = width as usize;
                    for _ in content_len..available_width.saturating_sub(2) {
                        print!(" ");
                    }
                    println!(" │");
                }
                
                // Draw bottom border
                stdout().execute(crossterm::cursor::MoveTo(0, start_row + box_height as u16 - 1))?;
                print!("╰");
                for _ in 1..width-1 {
                    print!("─");
                }
                println!("╯");
                
                stdout().flush()?;
                needs_redraw = false;
            }
            
            // Calculate cursor position
            cursor_line = 0;
            cursor_col = 0;
            
            for ch in buffer.chars().take(cursor_pos) {
                if ch == '\n' {
                    cursor_line += 1;
                    cursor_col = 0;
                } else {
                    cursor_col += 1;
                }
            }
            
            // Position cursor for input
            let cursor_x = 4 + cursor_col as u16;
            let cursor_y = start_row + 1 + cursor_line as u16;
            stdout().execute(crossterm::cursor::MoveTo(cursor_x, cursor_y))?;
            
            // Handle input with shorter timeout to catch resize events faster
            if event::poll(std::time::Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key_event) => {
                        match (key_event.code, key_event.modifiers) {
                            (KeyCode::Enter, KeyModifiers::NONE) => {
                                crossterm::terminal::disable_raw_mode()?;
                                // Position cursor after the textbox
                                let lines_count = buffer.split('\n').count().max(1);
                                let final_row = start_row + lines_count as u16 + 2;
                                stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
                                return Ok(buffer);
                            }
                            (KeyCode::Enter, KeyModifiers::ALT) => {
                                buffer.insert(cursor_pos, '\n');
                                cursor_pos += 1;
                                needs_redraw = true;
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                crossterm::terminal::disable_raw_mode()?;
                                let lines_count = buffer.split('\n').count().max(1);
                                let final_row = start_row + lines_count as u16 + 2;
                                stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
                                return Ok(String::new());
                            }
                            (KeyCode::Char(c), _) => {
                                buffer.insert(cursor_pos, c);
                                cursor_pos += 1;
                                needs_redraw = true;
                            }
                            (KeyCode::Backspace, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                    buffer.remove(cursor_pos);
                                    needs_redraw = true;
                                }
                            }
                            (KeyCode::Left, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                }
                            }
                            (KeyCode::Right, _) => {
                                if cursor_pos < buffer.len() {
                                    cursor_pos += 1;
                                }
                            }
                            (KeyCode::Up, _) => {
                                // Move cursor up a line
                                let mut new_pos = 0;
                                let mut line = 0;
                                let mut col = 0;
                                
                                for (i, ch) in buffer.chars().enumerate() {
                                    if line == cursor_line - 1 && col == cursor_col {
                                        new_pos = i;
                                        break;
                                    }
                                    if ch == '\n' {
                                        if line == cursor_line - 1 && col < cursor_col {
                                            new_pos = i;
                                            break;
                                        }
                                        line += 1;
                                        col = 0;
                                    } else {
                                        col += 1;
                                    }
                                }
                                
                                if cursor_line > 0 {
                                    cursor_pos = new_pos;
                                }
                            }
                            (KeyCode::Down, _) => {
                                // Move cursor down a line
                                let lines: Vec<&str> = buffer.split('\n').collect();
                                if cursor_line < lines.len() - 1 {
                                    let mut new_pos = 0;
                                    let mut line = 0;
                                    let mut col = 0;
                                    
                                    for (i, ch) in buffer.chars().enumerate() {
                                        if line == cursor_line + 1 && col == cursor_col {
                                            new_pos = i;
                                            break;
                                        }
                                        if ch == '\n' {
                                            if line == cursor_line + 1 && col < cursor_col {
                                                new_pos = i;
                                                break;
                                            }
                                            line += 1;
                                            col = 0;
                                        } else {
                                            col += 1;
                                        }
                                    }
                                    
                                    cursor_pos = new_pos;
                                }
                            }
                            (KeyCode::Esc, _) => {
                                crossterm::terminal::disable_raw_mode()?;
                                let lines_count = buffer.split('\n').count().max(1);
                                let final_row = start_row + lines_count as u16 + 2;
                                stdout().execute(crossterm::cursor::MoveTo(0, final_row))?;
                                return Ok(String::new());
                            }
                            _ => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        needs_redraw = true;
                    }
                    _ => {}
                }
            }
        }
    }
    
    pub fn run_at_position(start_row: u16, start_col: u16) -> io::Result<String> {
        crossterm::terminal::enable_raw_mode()?;
        
        let mut buffer = String::new();
        let mut cursor_pos = 0;
        let mut cursor_line = 0;
        let mut cursor_col = 0;
        
        let mut needs_redraw = true;
        let mut last_width = 0;
        let mut last_height = 0;
        
        loop {
            let (width, height) = terminal::size()?;
            
            // Check if terminal size changed
            if width != last_width || height != last_height {
                needs_redraw = true;
                last_width = width;
                last_height = height;
            }
            
            if needs_redraw {
                // Clear the entire screen to handle resize properly
                stdout().execute(terminal::Clear(ClearType::All))?;
                
                // Calculate box dimensions
                let lines: Vec<&str> = if buffer.is_empty() {
                    vec![""]
                } else {
                    buffer.split('\n').collect()
                };
                let box_height = lines.len().max(1) + 2; // +2 for borders
                
                // Move to starting position and draw top border
                stdout().execute(crossterm::cursor::MoveTo(start_col, start_row))?;
                print!("╭");
                for _ in 1..width.saturating_sub(start_col).saturating_sub(1) {
                    print!("─");
                }
                println!("╮");
                
                // Draw content lines
                for (i, line) in lines.iter().enumerate() {
                    stdout().execute(crossterm::cursor::MoveTo(start_col, start_row + 1 + i as u16))?;
                    if i == 0 {
                        print!("│ > ");
                    } else {
                        print!("│   ");
                    }
                    
                    // Show placeholder on first line if empty
                    if i == 0 && buffer.is_empty() {
                        print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                    } else {
                        print!("{}", line);
                    }
                    
                    // Fill the rest of the line
                    let content_len = if i == 0 && buffer.is_empty() { 
                        30 // length of placeholder + prompt
                    } else { 
                        line.len() + 4
                    };
                    let available_width = width.saturating_sub(start_col) as usize;
                    for _ in content_len..available_width.saturating_sub(2) {
                        print!(" ");
                    }
                    println!(" │");
                }
                
                // Draw bottom border
                stdout().execute(crossterm::cursor::MoveTo(start_col, start_row + box_height as u16 - 1))?;
                print!("╰");
                for _ in 1..width.saturating_sub(start_col).saturating_sub(1) {
                    print!("─");
                }
                println!("╯");
                
                stdout().flush()?;
                needs_redraw = false;
            }
            
            // Calculate cursor position
            cursor_line = 0;
            cursor_col = 0;
            
            for ch in buffer.chars().take(cursor_pos) {
                if ch == '\n' {
                    cursor_line += 1;
                    cursor_col = 0;
                } else {
                    cursor_col += 1;
                }
            }
            
            // Position cursor for input
            let cursor_x = start_col + 4 + cursor_col as u16;
            let cursor_y = start_row + 1 + cursor_line as u16;
            stdout().execute(crossterm::cursor::MoveTo(cursor_x, cursor_y))?;
            
            // Handle input with shorter timeout to catch resize events faster
            if event::poll(std::time::Duration::from_millis(16))? {  // ~60fps
                match event::read()? {
                    Event::Key(key_event) => {
                        match (key_event.code, key_event.modifiers) {
                            (KeyCode::Enter, KeyModifiers::NONE) => {
                                crossterm::terminal::disable_raw_mode()?;
                                println!();
                                return Ok(buffer);
                            }
                            (KeyCode::Enter, KeyModifiers::ALT) => {
                                buffer.insert(cursor_pos, '\n');
                                cursor_pos += 1;
                                needs_redraw = true;
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                crossterm::terminal::disable_raw_mode()?;
                                println!();
                                return Ok(String::new());
                            }
                            (KeyCode::Char(c), _) => {
                                buffer.insert(cursor_pos, c);
                                cursor_pos += 1;
                                needs_redraw = true;
                            }
                            (KeyCode::Backspace, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                    buffer.remove(cursor_pos);
                                    needs_redraw = true;
                                }
                            }
                            (KeyCode::Left, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                }
                            }
                            (KeyCode::Right, _) => {
                                if cursor_pos < buffer.len() {
                                    cursor_pos += 1;
                                }
                            }
                            (KeyCode::Up, _) => {
                                // Move cursor up a line
                                let mut new_pos = 0;
                                let mut line = 0;
                                let mut col = 0;
                                
                                for (i, ch) in buffer.chars().enumerate() {
                                    if line == cursor_line - 1 && col == cursor_col {
                                        new_pos = i;
                                        break;
                                    }
                                    if ch == '\n' {
                                        if line == cursor_line - 1 && col < cursor_col {
                                            new_pos = i;
                                            break;
                                        }
                                        line += 1;
                                        col = 0;
                                    } else {
                                        col += 1;
                                    }
                                }
                                
                                if cursor_line > 0 {
                                    cursor_pos = new_pos;
                                }
                            }
                            (KeyCode::Down, _) => {
                                // Move cursor down a line
                                let lines: Vec<&str> = buffer.split('\n').collect();
                                if cursor_line < lines.len() - 1 {
                                    let mut new_pos = 0;
                                    let mut line = 0;
                                    let mut col = 0;
                                    
                                    for (i, ch) in buffer.chars().enumerate() {
                                        if line == cursor_line + 1 && col == cursor_col {
                                            new_pos = i;
                                            break;
                                        }
                                        if ch == '\n' {
                                            if line == cursor_line + 1 && col < cursor_col {
                                                new_pos = i;
                                                break;
                                            }
                                            line += 1;
                                            col = 0;
                                        } else {
                                            col += 1;
                                        }
                                    }
                                    
                                    cursor_pos = new_pos;
                                }
                            }
                            (KeyCode::Esc, _) => {
                                crossterm::terminal::disable_raw_mode()?;
                                println!();
                                return Ok(String::new());
                            }
                            _ => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        needs_redraw = true;
                    }
                    _ => {}
                }
            }
        }
    }
    
    pub fn run_centered() -> io::Result<String> {
        crossterm::terminal::enable_raw_mode()?;
        
        let mut buffer = String::new();
        let mut cursor_pos = 0;
        let mut cursor_line = 0;
        let mut cursor_col = 0;
        
        let mut needs_redraw = true;
        let mut last_width = 0;
        let mut last_height = 0;
        
        loop {
            let (width, height) = terminal::size()?;
            
            // Calculate centered position
            let start_row = height.saturating_sub(5) / 2;
            let start_col = 0;
            
            // Check if terminal size changed
            if width != last_width || height != last_height {
                needs_redraw = true;
                last_width = width;
                last_height = height;
            }
            
            if needs_redraw {
                // Clear the entire screen to handle resize properly
                stdout().execute(terminal::Clear(ClearType::All))?;
                
                // Calculate box dimensions
                let lines: Vec<&str> = if buffer.is_empty() {
                    vec![""]
                } else {
                    buffer.split('\n').collect()
                };
                let box_height = lines.len().max(1) + 2; // +2 for borders
                
                // Move to starting position and draw top border
                stdout().execute(crossterm::cursor::MoveTo(start_col, start_row))?;
                print!("╭");
                for _ in 1..width.saturating_sub(start_col).saturating_sub(1) {
                    print!("─");
                }
                println!("╮");
                
                // Draw content lines
                for (i, line) in lines.iter().enumerate() {
                    stdout().execute(crossterm::cursor::MoveTo(start_col, start_row + 1 + i as u16))?;
                    if i == 0 {
                        print!("│ > ");
                    } else {
                        print!("│   ");
                    }
                    
                    // Show placeholder on first line if empty
                    if i == 0 && buffer.is_empty() {
                        print!("\x1b[90mTry \"fix typecheck errors\"\x1b[0m");
                    } else {
                        print!("{}", line);
                    }
                    
                    // Fill the rest of the line
                    let content_len = if i == 0 && buffer.is_empty() { 
                        30 // length of placeholder + prompt
                    } else { 
                        line.len() + 4
                    };
                    let available_width = width.saturating_sub(start_col) as usize;
                    for _ in content_len..available_width.saturating_sub(2) {
                        print!(" ");
                    }
                    println!(" │");
                }
                
                // Draw bottom border
                stdout().execute(crossterm::cursor::MoveTo(start_col, start_row + box_height as u16 - 1))?;
                print!("╰");
                for _ in 1..width.saturating_sub(start_col).saturating_sub(1) {
                    print!("─");
                }
                println!("╯");
                
                stdout().flush()?;
                needs_redraw = false;
            }
            
            // Calculate cursor position
            cursor_line = 0;
            cursor_col = 0;
            
            for ch in buffer.chars().take(cursor_pos) {
                if ch == '\n' {
                    cursor_line += 1;
                    cursor_col = 0;
                } else {
                    cursor_col += 1;
                }
            }
            
            // Position cursor for input
            let start_row = height.saturating_sub(5) / 2; // Recalculate for current height
            let cursor_x = start_col + 4 + cursor_col as u16;
            let cursor_y = start_row + 1 + cursor_line as u16;
            stdout().execute(crossterm::cursor::MoveTo(cursor_x, cursor_y))?;
            
            // Handle input with shorter timeout to catch resize events faster
            if event::poll(std::time::Duration::from_millis(16))? {
                match event::read()? {
                    Event::Key(key_event) => {
                        match (key_event.code, key_event.modifiers) {
                            (KeyCode::Enter, KeyModifiers::NONE) => {
                                crossterm::terminal::disable_raw_mode()?;
                                println!();
                                return Ok(buffer);
                            }
                            (KeyCode::Enter, KeyModifiers::ALT) => {
                                buffer.insert(cursor_pos, '\n');
                                cursor_pos += 1;
                                needs_redraw = true;
                            }
                            (KeyCode::Char('c'), KeyModifiers::CONTROL) => {
                                crossterm::terminal::disable_raw_mode()?;
                                println!();
                                return Ok(String::new());
                            }
                            (KeyCode::Char(c), _) => {
                                buffer.insert(cursor_pos, c);
                                cursor_pos += 1;
                                needs_redraw = true;
                            }
                            (KeyCode::Backspace, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                    buffer.remove(cursor_pos);
                                    needs_redraw = true;
                                }
                            }
                            (KeyCode::Left, _) => {
                                if cursor_pos > 0 {
                                    cursor_pos -= 1;
                                }
                            }
                            (KeyCode::Right, _) => {
                                if cursor_pos < buffer.len() {
                                    cursor_pos += 1;
                                }
                            }
                            (KeyCode::Up, _) => {
                                // Move cursor up a line
                                let mut new_pos = 0;
                                let mut line = 0;
                                let mut col = 0;
                                
                                for (i, ch) in buffer.chars().enumerate() {
                                    if line == cursor_line - 1 && col == cursor_col {
                                        new_pos = i;
                                        break;
                                    }
                                    if ch == '\n' {
                                        if line == cursor_line - 1 && col < cursor_col {
                                            new_pos = i;
                                            break;
                                        }
                                        line += 1;
                                        col = 0;
                                    } else {
                                        col += 1;
                                    }
                                }
                                
                                if cursor_line > 0 {
                                    cursor_pos = new_pos;
                                }
                            }
                            (KeyCode::Down, _) => {
                                // Move cursor down a line
                                let lines: Vec<&str> = buffer.split('\n').collect();
                                if cursor_line < lines.len() - 1 {
                                    let mut new_pos = 0;
                                    let mut line = 0;
                                    let mut col = 0;
                                    
                                    for (i, ch) in buffer.chars().enumerate() {
                                        if line == cursor_line + 1 && col == cursor_col {
                                            new_pos = i;
                                            break;
                                        }
                                        if ch == '\n' {
                                            if line == cursor_line + 1 && col < cursor_col {
                                                new_pos = i;
                                                break;
                                            }
                                            line += 1;
                                            col = 0;
                                        } else {
                                            col += 1;
                                        }
                                    }
                                    
                                    cursor_pos = new_pos;
                                }
                            }
                            (KeyCode::Esc, _) => {
                                crossterm::terminal::disable_raw_mode()?;
                                println!();
                                return Ok(String::new());
                            }
                            _ => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        needs_redraw = true;
                    }
                    _ => {}
                }
            }
        }
    }
}