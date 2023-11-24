use std::{
    cmp,
    collections::{HashMap, HashSet},
    str::from_utf8,
};

use framework::{
    app, App, Canvas, Color, CursorIcon, KeyCode, KeyState, KeyboardInput, Position, Rect, Size,
    Widget, WidgetData,
};

use headless_editor::{
    parse_tokens,
    transaction::{EditAction, TransactingVirtualCursor, Transaction, TransactionManager},
    SimpleCursor, SimpleTextBuffer, TextBuffer, TokenTextBuffer, VirtualCursor,
};
use itertools::Itertools;
use serde::{de, Deserialize, Deserializer, Serialize};
use serde_json::json;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextPane<Cursor: VirtualCursor> {
    line_height: f32,
    offset: Position,
    cursor: Cursor,
    text_buffer: TokenTextBuffer<SimpleTextBuffer>,
    color_mapping: HashMap<usize, String>,
    max_line_length: Option<usize>,
}

// TODO: Got some weird token missing that refreshing state fixes

impl<Cursor: VirtualCursor> TextPane<Cursor> {
    pub fn new(contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
            cursor: Cursor::new(0, 0),
            text_buffer: TokenTextBuffer::new_with_contents(&contents),
            color_mapping: HashMap::new(),
            max_line_length: None,
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

    pub fn on_scroll(&mut self, x: f64, y: f64, width: f32, height: f32, y_margin: i32) {
        if self.max_line_length.is_none() {
            self.max_line_length = Some(self.text_buffer.max_line_length());
        }

        self.offset.x += x as f32;

        let character_width = 18;
        if let Some(max_line) = self.max_line_length {
            let max_width = character_width * max_line;
            if self.offset.x + width > max_width as f32 {
                self.offset.x = max_width as f32 - width;
            }
        }

        if self.offset.x < 0.0 {
            self.offset.x = 0.0;
        }

        // TODO: Handle x scrolling too far
        self.offset.y -= y as f32;

        let scroll_with_last_line_visible =
            self.number_of_lines()
                .saturating_sub(self.number_of_visible_lines(height)) as f32
                * self.line_height
                + y_margin as f32;

        if height - y_margin as f32 > self.number_of_lines() as f32 * self.line_height {
            self.offset.y = 0.0;
            return;
        }

        // TODO: Deal with margin properly

        if self.offset.y > scroll_with_last_line_visible {
            self.offset.y = scroll_with_last_line_visible;
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
    text_pane: TextPane<TransactingVirtualCursor<SimpleCursor>>,
    widget_data: WidgetData,
    edit_position: usize,
    #[serde(default)]
    file_path: String,
    x_margin: i32,
    y_margin: i32,
    selecting: bool,
    diagnostics: DiagnosticMessage,
    #[serde(skip)]
    transaction_pane: Option<Widget>,
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

#[derive(Serialize, Deserialize, Clone, Debug)]
struct LineCharacter {
    line: usize,
    character: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Range {
    start: LineCharacter,
    end: LineCharacter,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Href {
    href: String,
}

// TODO: Make this work for serialize and deserialize
#[derive(Serialize, Clone, Debug)]
#[repr(u8)]
enum Severity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

impl<'de> Deserialize<'de> for Severity {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let code = u8::deserialize(deserializer)?;
        match code {
            1 => Ok(Severity::Error),
            2 => Ok(Severity::Warning),
            3 => Ok(Severity::Information),
            4 => Ok(Severity::Hint),
            _ => Err(de::Error::custom("Invalid value")),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct Diagnostic {
    range: Range,
    severity: usize,
    code: String,
    #[serde(rename = "codeDescription")]
    code_description: Option<Href>,
    source: String,
    message: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct DiagnosticMessage {
    uri: String,
    diagnostics: Vec<Diagnostic>,
    version: Option<usize>,
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
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
    fn start(&mut self) {
        self.subscribe("tokens_with_version");
        self.subscribe("color_mapping_changed");
        self.subscribe("diagnostics");
    }

    fn get_initial_state(&self) -> String {
        let init_self = Self::init();
        serde_json::to_string(&init_self).unwrap()
    }

    fn draw(&mut self) {
        // I need a proper update function

        if let Some(transaction_pane) = &mut self.transaction_pane {
            if let Some(transaction_pane) = transaction_pane.as_any_mut().downcast_mut::<Self>() {
                transaction_pane.text_pane.text_buffer.set_contents(
                    Self::format_transactions(self.text_pane.cursor.get_transaction_manager())
                        .as_bytes(),
                );
            } else {
                println!("No downcast!");
            }
        }

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

        let diagnostic_lines = self
            .diagnostics
            .diagnostics
            .iter()
            .map(|x| x.range.start.line)
            .collect::<HashSet<usize>>();

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
            if diagnostic_lines.contains(&(line)) {
                canvas.set_color(&Color::parse_hex("#ff0000"));
            }
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
        self.text_pane.cursor.remove_empty_selection();
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
        }
        self.set_cursor_icon(CursorIcon::Text);
    }

    fn on_key(&mut self, input: KeyboardInput) {
        self.handle_key_press(input);
        self.handle_edits();
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        self.text_pane.on_scroll(
            x,
            y,
            self.widget_data.size.width,
            self.widget_data.size.height,
            self.y_margin,
        );
    }

    fn get_state<'a>(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    fn set_state(&mut self, state: String) {
        let value: Self = serde_json::from_str(&state).unwrap();
        *self = value;
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
        } else if kind == "diagnostics" {
            if let Ok(diagnostics) = serde_json::from_str::<DiagnosticMessage>(&event) {
                if diagnostics.uri == format!("file://{}", self.file_path)
                    && (diagnostics.version.is_none()
                        || self.diagnostics.version <= diagnostics.version)
                {
                    self.diagnostics = diagnostics;
                }
            } else {
                println!("Couldn't parse {}", event);
            }
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.widget_data.size = Size { width, height };
    }

    fn on_move(&mut self, x: f32, y: f32) {
        self.widget_data.position = Position { x, y };
    }

    fn get_position(&self) -> Position {
        self.widget_data.position
    }

    fn get_size(&self) -> Size {
        self.widget_data.size
    }
}

impl TextWidget {
    fn init() -> Self {
        Self {
            text_pane: TextPane::new(vec![], 30.0),
            widget_data: WidgetData {
                position: Position { x: 0.0, y: 0.0 },
                size: Size {
                    width: 600.0,
                    height: 600.0,
                },
            },
            x_margin: 0,
            y_margin: 87,
            file_path: "".to_string(),
            edit_position: 0,
            selecting: false,
            diagnostics: DiagnosticMessage {
                uri: "".to_string(),
                diagnostics: vec![],
                version: Some(0),
            },
            transaction_pane: None,
        }
    }

    fn handle_key_press(&mut self, input: KeyboardInput) {
        if !matches!(input.state, KeyState::Pressed) {
            return;
        }

        // Order matters here
        if input.modifiers.cmd && input.modifiers.shift && matches!(input.key_code, KeyCode::Z) {
            self.text_pane.cursor.redo(&mut self.text_pane.text_buffer);
            return;
        }
        if input.modifiers.cmd && matches!(input.key_code, KeyCode::Z) {
            self.text_pane.cursor.undo(&mut self.text_pane.text_buffer);
            return;
        }

        if input.modifiers.ctrl && matches!(input.key_code, KeyCode::E) {
            self.text_pane
                .cursor
                .end_of_line(&self.text_pane.text_buffer);
            return;
        }

        if input.modifiers.ctrl && matches!(input.key_code, KeyCode::A) {
            self.text_pane.cursor.start_of_line();
            return;
        }

        if input.modifiers.ctrl
            && input.modifiers.cmd
            && input.modifiers.option
            && matches!(input.key_code, KeyCode::T)
        {
            let mut data = self.widget_data.clone();
            data.position.x += data.size.width + 50.0;
            let contents = self
                .text_pane
                .cursor
                .get_transactions()
                .iter()
                .map(|x| format!("{:?}", x))
                .fold(String::new(), |a, b| a + &b + "\n")
                .into_bytes();

            // TODO: Make it easy to update the widget data
            self.transaction_pane = Some(self.create_widget(
                Box::new(Self {
                    text_pane: TextPane::new(contents, 30.0),
                    widget_data: data.clone(),
                    edit_position: 0,
                    file_path: "".to_string(),
                    x_margin: 30,
                    y_margin: 60,
                    selecting: false,
                    diagnostics: DiagnosticMessage {
                        uri: "".to_string(),
                        diagnostics: vec![],
                        version: None,
                    },
                    transaction_pane: None,
                }),
                data,
            ));
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
                .delete(&mut self.text_pane.text_buffer),
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

    fn handle_edits(&mut self) {
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
    }

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

        let cursor = self.text_pane.cursor.clone();
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

    fn format_transaction<Cursor: VirtualCursor>(
        transaction_pointer: Option<usize>,
        undo_pointer: Option<&usize>,
        start_index: usize,
        transactions: Vec<&Transaction<Cursor>>,
    ) -> String {
        let mut result = String::new();
        let mut categories: HashSet<&str> = HashSet::new();
        let mut range: ((usize, usize), (usize, usize)) = ((usize::MAX, usize::MAX), (0, 0));
        let mut index = start_index;
        let mut contains_undo_pointer = "";
        let mut contains_transaction_pointer = "";
        // TODO: Delete is backwards.
        // But that might be the right thing to do?
        // Need to think about it
        for transaction in transactions {
            if Some(&index) == undo_pointer {
                contains_undo_pointer = ">> ";
            }

            if Some(index) == transaction_pointer {
                contains_transaction_pointer = "> ";
            }

            match &transaction.action {
                EditAction::Insert((line, column), text) => {
                    let text = from_utf8(text).unwrap();
                    result += text;
                    categories.insert("insert");
                    if (*line, *column) < range.0 {
                        range.0 = (*line, *column);
                    }
                    if (*line, *column) > range.1 {
                        range.1 = (*line, *column);
                    }
                }
                EditAction::Delete(start, end, text) => {
                    let text = from_utf8(text).unwrap();
                    result += text;
                    categories.insert("delete");

                    if start < &range.0 {
                        range.0 = *start;
                    }
                    if end > &range.1 {
                        range.1 = *end;
                    }
                }
                _ => {}
            }
            index += 1;
        }

        format!(
            "{}{}{}: {}, {:?} - {:?}",
            contains_undo_pointer,
            contains_transaction_pointer,
            categories.iter().join(", "),
            result.replace('\n', "\\n").replace(' ', "<space>"),
            range.0,
            range.1
        )
    }

    fn format_transactions<Cursor: VirtualCursor>(
        transaction_manager: &TransactionManager<Cursor>,
    ) -> String {
        let mut result = String::new();
        let groups = transaction_manager
            .transactions
            .iter()
            .enumerate()
            .group_by(|(_index, x)| x.transaction_number);

        for (transaction_number, group) in &groups {
            let group = group.collect_vec();
            let start_index = group.first().unwrap().0;
            result += &format!(
                "{} {}\n",
                transaction_number,
                &Self::format_transaction(
                    transaction_manager.transaction_pointer,
                    transaction_manager.undo_pointer_stack.last(),
                    start_index,
                    group.iter().map(|x| x.1).collect_vec()
                )
            );
        }

        result
    }
}

app!(TextWidget);
