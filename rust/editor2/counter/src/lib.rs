use std::str::from_utf8;

use framework::{app, App, Canvas, Color, KeyCode, KeyState, KeyboardInput, Rect, Value};
use headless_editor::{
    parse_tokens, Cursor, SimpleTextBuffer, TextBuffer, Token, TokenTextBuffer, VirtualCursor,
};
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
    text_buffer: TokenTextBuffer<SimpleTextBuffer>,
    tokens: Vec<Token>,
}

impl TextPane {
    pub fn new(contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            contents: vec![],
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
            cursor: Cursor::new(0, 0),
            text_buffer: TokenTextBuffer::new_with_contents(&contents),
            tokens: vec![],
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

impl App for TextWidget {
    type State = TextWidget;

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
        if let Some(tokens) = self.try_get_value("tokens") {
            if let Value::Bytes(bytes) = serde_json::from_str::<Value>(&tokens).unwrap() {
                if let Ok(tokens) = serde_json::from_slice::<Vec<u64>>(&bytes) {
                    let tokens = parse_tokens(&tokens);
                    if self.text_pane.tokens != tokens {
                        self.text_pane.tokens = tokens;
                        self.text_pane
                            .text_buffer
                            .set_tokens(self.text_pane.tokens.clone());
                    }
                }
            }
        }

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
        let length_output = &format!(
            "({}, {}) length: {}",
            cursor.line(),
            cursor.column(),
            text_buffer.line_length(cursor.line())
        );
        canvas.draw_str(
            length_output,
            self.widget_data.size.width - length_output.len() as f32 * 18.0,
            self.widget_data.size.height - 40.0,
        );

        let fractional_offset = self.text_pane.fractional_line_offset();
        canvas.translate(
            30.0 - self.text_pane.offset.x,
            self.text_pane.line_height - fractional_offset + 20.0,
        );

        canvas.save();

        if !self.text_pane.tokens.is_empty() {
            for line in self.text_pane.text_buffer.decorated_lines(
                self.text_pane.lines_above_scroll(),
                self.text_pane
                    .number_of_visible_lines(self.widget_data.size.height),
            ) {
                let mut x = 0.0;
                for (text, token) in line {
                    let foreground = if let Some(token) = token {
                        Color::new(
                            1.0 / token.kind as f32,
                            1.0 / token.modifiers as f32,
                            0.0,
                            1.0,
                        )
                    } else {
                        Color::parse_hex("#aa9941")
                    };
                    canvas.set_color(&foreground);
                    let text = from_utf8(text).unwrap().to_string();
                    // let text = if let Some(token) = token {
                    //     format!("length: {}, start: {}, text: {}", text.len(), token.delta_start, text)
                    // } else {
                    //     format!("empty: {}, text: {}", text.len(), text)
                    // };
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

        // TODO: I actually want the new edits, not all of them.
        for edit in self.text_pane.text_buffer.drain_edits() {
            self.send_event(
                "text_change".to_string(),
                serde_json::ser::to_string(&edit.edit).unwrap(),
            );
        }

        // TODO: Send message about what changed
        // Listen to that message and send to lsp
        // Update tokens

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
        self.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        // TODO: hacky
        let old_contents = self.text_pane.contents.clone();
        *self = state;
        self.text_pane.contents = old_contents;
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.widget_data.size = Size { width, height };
    }
}

app!(TextWidget);
