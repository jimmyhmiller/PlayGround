use std::str::from_utf8;

use framework::{app, App, Canvas, Color, KeyCode, KeyState, KeyboardInput, Rect, Value};
use headless_editor::{Cursor, SimpleTextBuffer, TextBuffer, VirtualCursor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

fn serialize_text<S>(x: &Vec<u8>, s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(from_utf8(x).unwrap())
}

fn deserialize_text<'de, D>(d: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    Ok(s.into_bytes())
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TextPane {
    #[serde(serialize_with = "serialize_text")]
    #[serde(deserialize_with = "deserialize_text")]
    contents: Vec<u8>,
    line_height: f32,
    offset: Position,
    cursor: Cursor,
    text_buffer: SimpleTextBuffer,
}

impl TextPane {
    pub fn new(contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            contents: vec![],
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
            cursor: Cursor::new(0, 0),
            text_buffer: SimpleTextBuffer::new_with_contents(&contents),
        }
    }

    pub fn _set_contents(&mut self, contents: Vec<u8>) {
        self.contents = contents;
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
}

#[derive(Serialize, Deserialize, Clone)]
struct WidgetData {
    position: Position,
    size: Size,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

#[derive(Serialize, Deserialize, Clone)]
struct TextWidget {
    text_pane: TextPane,
    widget_data: WidgetData,
    edit_position: usize,
}

pub struct Token {
    pub delta_line: usize,
    pub delta_start: usize,
    pub length: usize,
    pub kind: usize,
    pub modifiers: usize,
}


pub struct TokenLineIter<'a> {
    current_position: usize,
    tokens: &'a [Token],
    empty_lines: usize,
}

impl<'a> Iterator for TokenLineIter<'a> 
{
    type Item = &'a [Token];
    fn next(&mut self) -> Option<Self::Item> {
        let original_position = self.current_position;

        if self.empty_lines > 0 {
            self.empty_lines -= 1;
            return Some(&[]);
        }
        while self.current_position < self.tokens.len() {
            let token = &self.tokens[self.current_position];
            if self.current_position != original_position && token.delta_line == 1 {
                self.empty_lines = 0;
                return Some(&self.tokens[original_position..self.current_position]);

            } else if self.current_position != original_position && token.delta_line > 1 {
                self.empty_lines = token.delta_line - 1;
                return Some(&self.tokens[original_position..self.current_position]);
            }
            self.current_position += 1;
        }
        if self.current_position != original_position {
            let line = &self.tokens[original_position..self.current_position];
            return Some(line);
        }
        None
    }
}

trait TokenLinerIterExt<'a> {
    fn token_lines(self) -> TokenLineIter<'a>;
}


impl <'a> TokenLinerIterExt<'a> for &'a [Token] {
    fn token_lines(self) -> TokenLineIter<'a> {
        TokenLineIter {
            current_position: 0,
            tokens: self,
            empty_lines: 0,
        }
    }
}

fn make_decorted_line<'a>(line: &'a[u8], tokens: &'a[Token]) -> Vec<(&'a[u8], Option<&'a Token>)> {
    let mut result = vec![];
    let mut current_position = 0;
    let mut last_end = 0;
    for (i, token) in tokens.iter().enumerate() {
        current_position += token.delta_start;
        if current_position > last_end {
            let non_token_range = last_end..current_position;
            result.push((&line[non_token_range], None));
        }
        let end = current_position + token.length;
        let token_range = current_position..end;
        result.push((&line[token_range], Some(token)));
        last_end = end;
    }
    if last_end < line.len() {
        let non_token_range = last_end..line.len();
        result.push((&line[non_token_range], None));
    }
    result
}

impl App for TextWidget {
    type State = TextPane;

    fn init() -> Self {
        let file = "/code/process-test/src/lib.rs";
        let contents = std::fs::read(file).unwrap();
        Self {
            text_pane: TextPane::new(contents, 30.0),
            widget_data: WidgetData {
                position: Position { x: 0.0, y: 0.0 },
                size: Size {
                    width: 600.0,
                    height: 600.0,
                },
            },
            edit_position: 0,
        }
    }

    fn draw(&mut self) {


        // TODO: Clean up
        // Only do this when I need to
        let tokens = if let Some(tokens) = self.try_get_value("tokens") {
            if let Value::Bytes(bytes) = serde_json::from_str::<Value>(&tokens).unwrap() {
                let tokens: Vec<u64> = serde_json::from_slice(&bytes).unwrap();
                let tokens = parse_tokens(&tokens);
                Some(tokens)
            } else {
                None
            }
        } else {
            None
        };


        let canvas = Canvas::new();

        let foreground = Color::parse_hex("#dc9941");
        let background = Color::parse_hex("#353f38");

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

        canvas.translate(self.widget_data.position.x, self.widget_data.position.y);

        let cursor = &self.text_pane.cursor;
        let text_buffer = &self.text_pane.text_buffer;

        canvas.set_color(&foreground);
        canvas.draw_str(
            &format!(
                "({}, {}) length: {}",
                cursor.line(),
                cursor.column(),
                text_buffer.line_length(cursor.line())
            ),
            300.0,
            500.0,
        );

        let fractional_offset = self.text_pane.fractional_line_offset();
        canvas.translate(
            30.0 - self.text_pane.offset.x,
            self.text_pane.line_height - fractional_offset + 20.0,
        );

        canvas.save();

        if let Some(tokens) = tokens {
            let token_iter = tokens.token_lines();
            let line_iter = self.text_pane.text_buffer.lines();
            let zipped = line_iter.zip(token_iter);
            for (line, tokens) in zipped
                .skip(self.text_pane.lines_above_scroll())
                .take(
                    self.text_pane
                        .number_of_visible_lines(self.widget_data.size.height),
                )
            {
                

                let mut x = 0.0;
                for (text, token) in make_decorted_line(line, tokens) {
                    let foreground = if let Some(token) = token {
                        Color::new( 1.0 / token.kind as f32, 1.0 / token.modifiers as f32, 0.0, 1.0)
                    } else {
                        Color::parse_hex("#aa9941")
                    };
                    canvas.set_color(&foreground);
                    let text = from_utf8(text).unwrap().to_string();
                    canvas.draw_str(&text, x, 0.0);
                    x += text.len() as f32 * 16.0;
                }
                canvas.translate(0.0, self.text_pane.line_height);
            }
        } else {
            for line in self
                .text_pane
                .text_buffer
                .lines()
                .skip(self.text_pane.lines_above_scroll())
                .take(
                    self.text_pane
                        .number_of_visible_lines(self.widget_data.size.height),
                )
            {
                canvas.set_color(&foreground);
                if let Ok(line) = from_utf8(line) {
                    canvas.draw_str(line, 0.0, 0.0);
                    canvas.translate(0.0, self.text_pane.line_height);
                }
            }
        }
        canvas.restore();

        let lines_above = self.text_pane.lines_above_scroll();
        let num_lines = self
            .text_pane
            .number_of_visible_lines(self.widget_data.size.height);
        let shown_lines = lines_above..lines_above + num_lines;

        if shown_lines.contains(&cursor.line()) {
            let nth_line = cursor.line() - lines_above;
            let cursor_position_pane_x = cursor.column() as f32 * 16.0;
            let cursor_position_pane_y = (nth_line as f32 - 1.0) * self.text_pane.line_height;

            canvas.set_color(&foreground);
            canvas.draw_rect(
                cursor_position_pane_x,
                cursor_position_pane_y,
                3.0,
                self.text_pane.line_height,
            );
        }

        canvas.restore();
    }

    fn on_click(&mut self, x: f32, y: f32) {
        // TODO: Need to handle margin here.
        let margin = 20;
        let lines_above = self.text_pane.lines_above_scroll();
        let line =
            (((y - margin as f32) / self.text_pane.line_height).ceil() as usize + lines_above) - 1;
        let char_width = 16.0;
        let column = (((x - margin as f32) / char_width).ceil() as usize)
            .saturating_sub((self.text_pane.offset.x / char_width) as usize)
            - 1;
        self.text_pane
            .cursor
            .move_to_bounded(line, column, &self.text_pane.text_buffer);
    }

    fn on_key(&mut self, input: KeyboardInput) {
        if !matches!(input.state, KeyState::Pressed) {
            return;
        }
        match input.key_code {
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
            _ => {}
        }
        if let Some(char) = input.to_char() {
            self.text_pane
                .cursor
                .handle_insert(&[char as u8], &mut self.text_pane.text_buffer);
        }

        match input.key_code {
            KeyCode::UpArrow => {
                if self.text_pane.cursor.line() == self.text_pane.lines_above_scroll() {
                    // round down to the fraction of a line so the whole text is visible
                    self.text_pane.offset.y -= self.text_pane.fractional_line_offset();
                } else if self.text_pane.cursor.line() < self.text_pane.lines_above_scroll() {
                    self.text_pane.offset.y -= self.text_pane.line_height;
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
        self.text_pane.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        // TODO: hacky
        // let old_contents = self.text_pane.contents.clone();
        // self.text_pane = state;
        // self.text_pane.contents = old_contents;
    }
}

impl From<&[u64]> for Token {
    fn from(chunk: &[u64]) -> Self {
        assert!(
            chunk.len() == 5,
            "Expected chunk to be of length 5, but was {}",
            chunk.len(),
        );
        Token {
            delta_line: chunk[0] as usize,
            delta_start: chunk[1] as usize,
            length: chunk[2] as usize,
            kind: chunk[3] as usize,
            modifiers: chunk[4] as usize,
            
        }
    }
}

fn parse_tokens(tokens: &[u64]) -> Vec<Token> {
    tokens.chunks(5).map(|chunk| Token::from(chunk)).collect()
}

app!(TextWidget);
