use std::{cmp, collections::HashMap, str::from_utf8};

use framework::{
    app, App, Canvas, Color, CursorIcon, KeyCode, KeyState, KeyboardInput, Position, Rect, Size,
    WidgetData,
};
use headless_editor::{
    parse_tokens, Cursor, SimpleTextBuffer, TextBuffer, Token, TokenTextBuffer, VirtualCursor,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextPane {
    line_height: f32,
    offset: Position,
    cursor: Cursor,
    text_buffer: TokenTextBuffer<SimpleTextBuffer>,
    color_mapping: HashMap<usize, String>,
}

// TODO: Got some weird token missing that refreshing state fixes

impl TextPane {
    pub fn new(contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
            cursor: Cursor::new(0, 0),
            text_buffer: TokenTextBuffer::new_with_contents(&contents),
            color_mapping: HashMap::new(),
        }
    }

    fn lines_above_scroll(&self) -> usize {
        (self.offset.y / self.line_height).floor() as usize
    }

    // TODO: Deal with margin!
    fn number_of_visible_lines(&self, height: f32) -> usize {
        ((height - 40.0) / self.line_height).ceil() as usize + 1
    }

    fn number_of_lines(&self) -> usize {
        self.text_buffer.line_count()
    }

    pub fn on_scroll(&mut self, x: f64, y: f64, height: f32) {
        // this is all terribly wrong. I don't get the bounds correctly.
        // Need to deal with that.

        self.offset.x += x as f32;
        if self.offset.x < 0.0 {
            self.offset.x = 0.0;
        }
        // TODO: Handle x scrolling too far
        self.offset.y -= y as f32;

        let scroll_with_last_line_visible =
            self.number_of_lines()
                .saturating_sub(self.number_of_visible_lines(height)) as f32
                * self.line_height;

        // TODO: Deal with margin properly

        if self.offset.y > scroll_with_last_line_visible + 40.0 {
            self.offset.y = scroll_with_last_line_visible + 40.0;
        }

        if self.offset.y < 0.0 {
            self.offset.y = 0.0;
        }
    }

    fn fractional_line_offset(&self) -> f32 {
        self.offset.y % self.line_height
    }

    fn set_color_mapping(&mut self, mapping: HashMap<usize, String>) {
        self.color_mapping = mapping;
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct TextWidget {
    text_pane: TextPane,
    widget_data: WidgetData,
    edit_position: usize,
    #[serde(default)]
    file_path: String,
    staged_tokens: Vec<Token>,
    x_margin: i32,
    selecting: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct EditWithPath {
    edit: headless_editor::Edit,
    path: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct MultiEditWithPath {
    version: usize,
    edits: Vec<headless_editor::Edit>,
    path: String,
}

#[derive(Clone, Deserialize, Serialize)]
struct TokensWithVersion {
    tokens: Vec<u64>,
    version: usize,
    path: String,
}

#[derive(Serialize, Deserialize, Clone)]
struct Tokens {
    path: String,
    tokens: Vec<u64>,
}

fn get_last_three_segments(path: &str) -> Option<String> {
    use std::path::Path;
    let path = Path::new(path);
    let mut components = path.components().rev();
    let file_name = components.next()?;
    let folder_name = components.next()?;
    let parent_folder = components.next()?;
    Some(format!(
        "{}/{}/{}",
        parent_folder.as_os_str().to_string_lossy(),
        folder_name.as_os_str().to_string_lossy(),
        file_name.as_os_str().to_string_lossy()
    ))
}

impl App for TextWidget {
    type State = TextWidget;

    fn init() -> Self {
        let me = Self {
            text_pane: TextPane::new(vec![], 30.0),
            widget_data: WidgetData {
                position: Position { x: 0.0, y: 0.0 },
                size: Size {
                    width: 600.0,
                    height: 600.0,
                },
            },
            x_margin: 0,
            file_path: "".to_string(),
            edit_position: 0,
            staged_tokens: vec![],
            selecting: false,
        };
        me.subscribe("tokens_with_version");
        me.subscribe("color_mapping_changed");
        me
    }

    fn draw(&mut self) {
        // TODO: I'm not showing fractional lines like I should
        let mut canvas = Canvas::new();

        let foreground = Color::parse_hex("#dc9941");
        let background = Color::parse_hex("#353f38");

        canvas.set_color(&foreground);

        // self.draw_debug(&mut canvas);

        let bounding_rect = Rect::new(
            0.0,
            0.0,
            self.widget_data.size.width,
            self.widget_data.size.height,
        );

        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);

        canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));

        let cursor = &self.text_pane.cursor;
        let text_buffer = &self.text_pane.text_buffer;

        canvas.set_color(&foreground);
        let length_output = &format!(
            "({}, {}) length: {}",
            cursor.line(),
            cursor.column(),
            text_buffer.line_length(cursor.line())
        ); 

        // canvas.draw_str(&format!("{:?}", self.text_pane.cursor.selection()), 700.0, 700.0);

        canvas.draw_str(
            length_output,
            self.widget_data.size.width - length_output.len() as f32 * 18.0,
            self.widget_data.size.height - 40.0,
        );

        if let Some(file_and_folder) = get_last_three_segments(&self.file_path) {
            canvas.draw_str(&file_and_folder, 20.0, 48.0);
        }

        canvas.translate(0.0, 84.0);
        canvas.clip_rect(bounding_rect);
        let fractional_offset = self.text_pane.fractional_line_offset();
        self.x_margin = 30;
        canvas.translate(
            self.x_margin as f32 - self.text_pane.offset.x,
            self.text_pane.line_height - fractional_offset,
        );

        canvas.save();
        let number_lines = self.text_pane.number_of_lines();
        let number_of_digits = number_lines.to_string().len();
        let current_line = self.text_pane.lines_above_scroll();
        let max_line = current_line
            + self
                .text_pane
                .number_of_visible_lines(self.widget_data.size.height);
        let max_line = max_line.min(number_lines);
        for line in current_line..max_line {
            canvas.set_color(&Color::parse_hex("#83CDA1"));
            let line_number = format!("{:width$}", line + 1, width = number_of_digits);
            canvas.draw_str(&line_number, 0.0, 0.0);
            canvas.translate(0.0, self.text_pane.line_height);
        }
        canvas.restore();

        canvas.save();
        let line_number_margin = (number_of_digits as f32 + 4.0 * 16.0) + 16.0;
        self.x_margin += line_number_margin as i32;
        canvas.translate(line_number_margin, 0.0);

        let lines_above = self.text_pane.lines_above_scroll();
        let num_lines = self
            .text_pane
            .number_of_visible_lines(self.widget_data.size.height);
        let shown_lines = lines_above..lines_above + num_lines;

        if shown_lines.contains(&cursor.line()) {
            let nth_line = cursor.line() - lines_above;
            let cursor_position_pane_x = cursor.column() as f32 * 16.0;
            let cursor_position_pane_y = (nth_line as f32 - 1.0) * self.text_pane.line_height + 6.0;

            canvas.set_color(&foreground);
            canvas.draw_rect(
                cursor_position_pane_x,
                cursor_position_pane_y,
                3.0,
                self.text_pane.line_height,
            );
        }

        let mut current_line = self.text_pane.lines_above_scroll();
        for line in self.text_pane.text_buffer.decorated_lines(
            current_line,
            self.text_pane
                .number_of_visible_lines(self.widget_data.size.height),
        ) {
            self.draw_selection(current_line, &mut canvas);
            let mut x = 0.0;
            for (text, token) in line {
                let foreground = Color::parse_hex(
                    token
                        .and_then(|token| self.text_pane.color_mapping.get(&token.kind))
                        .unwrap_or(&"#aa9941".to_string()),
                );
                canvas.set_color(&foreground);
                let text = from_utf8(text).unwrap().to_string();
                canvas.draw_str(&text, x, 0.0);
                x += text.len() as f32 * 16.0;
            }
            canvas.translate(0.0, self.text_pane.line_height);
            current_line += 1;
        }

        canvas.restore();

        canvas.restore();
    }

    fn on_click(&mut self, x: f32, y: f32) {
        if self.text_pane.text_buffer.tokens.is_empty() {
            self.send_open_file();
        }

        let (line, column) = self.find_cursor_text_position(x, y);

        self.text_pane
            .cursor
            .move_to_bounded(line, column, &self.text_pane.text_buffer);

        self.text_pane.cursor.set_selection_ordered(None);
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) {
        let (line, column) = self.find_cursor_text_position(x, y);

        self.text_pane
            .cursor
            .move_to_bounded(line, column, &self.text_pane.text_buffer);

        let line = self.text_pane.cursor.line();
        let column = self.text_pane.cursor.column();
        // TODO: I'm getting false positive selections
        self.text_pane
            .cursor
            .set_selection(Some(((line, column), (line, column))));
        self.selecting = true;
    }

    fn on_mouse_up(&mut self, _x: f32, _y: f32) {
        self.selecting = false;
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) {
        if x_diff.abs() < 0.001 && y_diff.abs() < 0.001 {
            return;
        }
        if self.selecting {
            if let Some(_current_selection) = self.text_pane.cursor.selection() {
                let (line, column) = self.find_cursor_text_position(x, y);
                // TODO: I need to find this in text space bounded
                let bounded_cursor = self.text_pane.cursor.nearest_text_position(
                    line,
                    column,
                    &self.text_pane.text_buffer,
                );
                let line = bounded_cursor.line();
                let column = bounded_cursor.column();
                self.text_pane.cursor.set_selection_movement((line, column));
            }
            self.set_cursor_icon(CursorIcon::Text);
        }
    }

    fn on_key(&mut self, input: KeyboardInput) {
        if !matches!(input.state, KeyState::Pressed) {
            return;
        }
        match input.key_code {
            KeyCode::Tab => self
                .text_pane
                .cursor
                .handle_insert("    ".as_bytes(), &mut self.text_pane.text_buffer),
            KeyCode::LeftArrow => self.text_pane.cursor.move_left(&self.text_pane.text_buffer),
            KeyCode::RightArrow => self
                .text_pane
                .cursor
                .move_right(&self.text_pane.text_buffer),
            KeyCode::UpArrow => self.text_pane.cursor.move_up(&self.text_pane.text_buffer),
            KeyCode::DownArrow => self.text_pane.cursor.move_down(&self.text_pane.text_buffer),
            KeyCode::BackSpace => self
                .text_pane
                .cursor
                .delete_char(&mut self.text_pane.text_buffer),
            KeyCode::S => {
                if input.modifiers.cmd {
                    self.save_file(
                        self.file_path.clone(),
                        from_utf8(self.text_pane.text_buffer.contents())
                            .unwrap()
                            .to_string(),
                    );
                    return;
                }
            }
            _ => {}
        }
        if let Some(char) = input.to_char() {
            self.text_pane
                .cursor
                .handle_insert(&[char as u8], &mut self.text_pane.text_buffer);
        }

        let edits = self.text_pane.text_buffer.drain_edits();
        if !edits.is_empty() {
            self.send_event(
                "text_change_multi",
                serde_json::ser::to_string(&MultiEditWithPath {
                    version: self.text_pane.text_buffer.document_version,
                    edits: edits.iter().map(|x| x.edit.clone()).collect(),
                    path: self.file_path.clone(),
                })
                .unwrap(),
            );
        }

        // for edit in edits.clone() {
        //     self.send_event(
        //         "text_change",
        //         serde_json::ser::to_string(&EditWithPath {
        //             edit: edit.edit,
        //             path: self.file_path.clone(),
        //         })
        //         .unwrap(),
        //     );
        // }

        match input.key_code {
            KeyCode::UpArrow => {
                match self
                    .text_pane
                    .cursor
                    .line()
                    .cmp(&self.text_pane.lines_above_scroll())
                {
                    cmp::Ordering::Equal => {
                        // round down to the fraction of a line so the whole text is visible
                        self.text_pane.offset.y -= self.text_pane.fractional_line_offset();
                    }
                    cmp::Ordering::Less => {
                        self.text_pane.offset.y -= self.text_pane.line_height;
                    }
                    cmp::Ordering::Greater => {}
                }
            }
            KeyCode::DownArrow => {
                let drawable_area_height = self.widget_data.size.height - 40.0;
                let logical_line =
                    self.text_pane.cursor.line() - self.text_pane.lines_above_scroll();
                let line_top = logical_line as f32 * self.text_pane.line_height
                    - self.text_pane.fractional_line_offset();
                let diff = drawable_area_height - line_top;

                if diff > 0.0 && diff < self.text_pane.line_height {
                    // not quite right yet
                    self.text_pane.offset.y += self.text_pane.line_height - diff;
                } else if self.text_pane.cursor.line() + 1
                    >= self.text_pane.lines_above_scroll()
                        + self
                            .text_pane
                            .number_of_visible_lines(self.widget_data.size.height)
                {
                    self.text_pane.offset.y += self.text_pane.line_height;
                }
            }
            _ => {}
        }
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        self.text_pane.on_scroll(x, y, self.widget_data.size.height);
    }

    fn get_state(&self) -> Self::State {
        self.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        *self = state;
        if !self.file_path.is_empty() {
            let file = &self.file_path;
            let contents = std::fs::read(file);
            match contents {
                Ok(contents) => {
                    self.send_open_file();
                    self.text_pane.text_buffer.set_contents(&contents);
                }
                Err(e) => {
                    println!("Error reading file: {}\n\n{}", file, e);
                }
            }
        }
    }

    fn on_event(&mut self, kind: String, event: String) {
        if kind == "tokens_with_version" {
            if let Ok(tokens) = serde_json::from_str::<TokensWithVersion>(&event) {
                if tokens.path != self.file_path {
                    return;
                }
                if tokens.version != self.text_pane.text_buffer.document_version {
                    println!(
                        "version mismatch tokens: {}  pane: {}",
                        tokens.version, self.text_pane.text_buffer.document_version
                    );
                    return;
                }
                let tokens = parse_tokens(&tokens.tokens);
                if !tokens.is_empty() {
                    // self.staged_tokens = tokens.clone();
                    self.text_pane.text_buffer.set_tokens(tokens);
                }
            } else {
                println!("Error parsing tokens: {}", event);
            }
        } else if kind == "color_mapping_changed" {
            if let Ok(mapping) =
                serde_json::from_str::<HashMap<usize, String>>(from_utf8(event.as_bytes()).unwrap())
            {
                self.text_pane.set_color_mapping(mapping);
            }
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.widget_data.size = Size { width, height };
    }

    fn on_move(&mut self, x: f32, y: f32) {
        self.widget_data.position = Position { x, y };
    }
}

impl TextWidget {
    fn send_open_file(&mut self) {
        self.send_event(
            "lith/open-file",
            json!({
                "version": self.text_pane.text_buffer.document_version,
                "path": self.file_path,
            })
            .to_string(),
        );
    }

    fn find_cursor_text_position(&mut self, x: f32, y: f32) -> (usize, usize) {
        // TODO: Need to handle margin here.
        let x_margin = self.x_margin;
        let y_margin = 80.0;
        let lines_above = self.text_pane.lines_above_scroll();
        let line = (((y - y_margin as f32) / self.text_pane.line_height).ceil() as usize
            + lines_above)
            .saturating_sub(1);
        let char_width = 16.0;
        let column = (((x + self.text_pane.offset.x - x_margin as f32) / char_width).ceil()
            as usize)
            .saturating_sub(1);
        (line, column)
    }

    #[allow(unused)]
    fn draw_debug(&mut self, canvas: &mut Canvas) {
        let foreground = Color::parse_hex("#dc9941");

        let cursor = self.text_pane.cursor;
        let current_token_window = self
            .text_pane
            .text_buffer
            .find_token(cursor.line(), cursor.column());

        let token_output: &String = &format!("{:#?}", current_token_window);
        let context = 10;
        let context_window = self
            .text_pane
            .text_buffer
            .tokens
            .iter()
            .enumerate()
            .skip(current_token_window.index.saturating_sub(context))
            .take(context * 2 + 1);

        for (i, (index, token)) in context_window.enumerate() {
            if index == current_token_window.index {
                canvas.set_color(&Color::parse_hex("#ffffff"));
            } else {
                canvas.set_color(&foreground);
            }
            canvas.draw_str(&format!("{:?}", token), 0.0, -700.0 + i as f32 * 32.0);
        }

        for (i, line) in token_output.lines().enumerate() {
            canvas.draw_str(line, -600.0, -400.0 + i as f32 * 32.0);
        }

        let x = 20;
        let last_x_token_actions = self
            .text_pane
            .text_buffer
            .token_actions
            .len()
            .saturating_sub(x);
        for (i, action) in self
            .text_pane
            .text_buffer
            .token_actions
            .iter()
            .skip(last_x_token_actions)
            .enumerate()
        {
            let action = format!("{:?}", action);
            canvas.draw_str(&action, 1200.0, 0.0 + i as f32 * 32.0);
        }
    }

    // TODO: Doesn't work on first line
    fn draw_selection(&self, line: usize, canvas: &mut Canvas) {
        canvas.save();
        canvas.translate(0.0, -30.0);

        canvas.set_color(&Color::parse_hex("#83CDA1").with_alpha(0.2));
        if let Some((start, end)) = self.text_pane.cursor.selection() {
            // println!("line: {}, start: {}", line, start.0);
            if line > start.0 && line < end.0 {
                let line_length = self.text_pane.text_buffer.line_length(line).max(1);
                canvas.draw_rect(0.0, 0.0, (line_length * 16) as f32, 30.0);
            } else if line == start.0 {
                let line_offset = start.1;
                let line_length = self.text_pane.text_buffer.line_length(line).max(1);
                let selection_length = if start.0 == end.0 {
                    end.1 - start.1
                } else {
                    line_length.saturating_sub(line_offset)
                };
                let draw_length = selection_length.min(line_length);
                canvas.draw_rect(
                    (line_offset * 16) as f32,
                    0.0,
                    (draw_length * 16) as f32,
                    30.0,
                );
            } else if line == end.0 {
                let line_length = self.text_pane.text_buffer.line_length(line).max(1);
                let selection_length = if start.0 == end.0 {
                    end.1.saturating_sub(start.1)
                } else {
                    end.1
                };
                let draw_length = selection_length.min(line_length);
                canvas.draw_rect(0.0, 0.0, (draw_length * 16) as f32, 30.0);
            }
        }
        canvas.restore();
    }
}

app!(TextWidget);
