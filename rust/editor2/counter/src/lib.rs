use std::str::from_utf8;

use framework::{App, Canvas, Color, Rect, KeyState, KeyCode};
use headless_editor::{Cursor, TextBuffer, SimpleTextBuffer, VirtualCursor};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
mod framework;

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



impl App for TextWidget {
    type State = TextPane;

    fn init() -> Self {
        Self {
            text_pane: TextPane::new(vec![], 30.0),
            widget_data: WidgetData {
                position: Position { x: 0.0, y: 0.0 },
                size: Size { width: 600.0, height: 600.0 },
            },
            edit_position: 0,
        }
    }

    fn draw(&mut self) {

        // let process_id = self.start_process("test-process".to_string());
        // self.send_message(process_id, "SENDING!".to_string());
        // println!("Got Message {}", self.recieve_last_message(process_id));



        let canvas = Canvas::new();


        let foreground = Color::parse_hex("#62b4a6");
        let background = Color::parse_hex("#530922");

        let bounding_rect = Rect::new(0.0, 0.0, self.widget_data.size.width, self.widget_data.size.height);

        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);
        // TODO: deal with fonts
        // let font = Font::new(
        //     Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
        //     32.0,
        // );

        canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));

        canvas.translate(self.widget_data.position.x, self.widget_data.position.y);

        let cursor = &self.text_pane.cursor;
        let text_buffer = &self.text_pane.text_buffer;

        canvas.set_color(&foreground);
        canvas.draw_str(&format!("({}, {}) length: {}", cursor.line(), cursor.column(), text_buffer.line_length(cursor.line())), 300.0, 500.0);

        let fractional_offset = self.text_pane.fractional_line_offset();
        canvas.translate(
            30.0 - self.text_pane.offset.x,
            self.text_pane.line_height - fractional_offset + 20.0,
        );

        canvas.save();

        for line in self.text_pane.text_buffer.lines()
                .skip(self.text_pane.lines_above_scroll())
                .take(self.text_pane.number_of_visible_lines(self.widget_data.size.height)) {
            canvas.draw_str(from_utf8(line).unwrap(), 0.0, 0.0);
            canvas.translate(0.0, self.text_pane.line_height);
        }
        canvas.restore();

        let lines_above = self.text_pane.lines_above_scroll();
        let num_lines = self.text_pane.number_of_visible_lines(self.widget_data.size.height);
        let shown_lines = lines_above..lines_above+num_lines;

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




        // let line_height = 30.0;
        // canvas.set_color(foreground);
        // for line in text.split("\\n") {
        //     canvas.draw_str(line, 0.0, 0.0);
        //     canvas.translate(0.0, line_height);
        // }

        canvas.restore();
        // canvas.draw_rect(0.0, 0.0, 240 as f32, 100 as f32);
        // canvas.draw_str(&format!("Count: {}", self.count), 40.0, 50.0);
    }


    fn on_click(&mut self, x: f32, y: f32) {
        // TODO: Remove
        if x == 0.0 && y == 0.0 {
            println!("Click");
        } else {
        // TODO: Need to handle margin here.
            let margin = 20;
            let lines_above = self.text_pane.lines_above_scroll();
            let line = (((y - margin as f32) / self.text_pane.line_height).ceil() as usize + lines_above) - 1;
            let char_width = 16.0;
            let column = (((x - margin as f32) / char_width).ceil() as usize).saturating_sub((self.text_pane.offset.x / char_width) as usize) - 1;
            self.text_pane.cursor.move_to_bounded(line, column, &self.text_pane.text_buffer);
        }

    }

    fn on_key(&mut self, input: KeyboardInput) {

        println!("Key: {:?}", input);
        if !matches!(input.state, KeyState::Pressed) {
            return;
        }
        match input.key_code {
            KeyCode::LeftArrow => self.text_pane.cursor.move_left(&self.text_pane.text_buffer),
            KeyCode::RightArrow => self.text_pane.cursor.move_right(&self.text_pane.text_buffer),
            KeyCode::UpArrow => self.text_pane.cursor.move_up(&self.text_pane.text_buffer),
            KeyCode::DownArrow => self.text_pane.cursor.move_down(&self.text_pane.text_buffer),
            KeyCode::BackSpace => self.text_pane.cursor.delete_char(&mut self.text_pane.text_buffer),
            _ => {}
        }
        if let Some(char) = input.to_char() {
            self.text_pane.cursor.handle_insert(&[char as u8], &mut self.text_pane.text_buffer);
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
                let logical_line = self.text_pane.cursor.line() - self.text_pane.lines_above_scroll();
                let line_top = logical_line as f32 * self.text_pane.line_height - self.text_pane.fractional_line_offset();
                let diff = drawable_area_height - line_top;

                if diff > 0.0 && diff < self.text_pane.line_height {
                    // not quite right yet
                    self.text_pane.offset.y += self.text_pane.line_height - diff;
                }  else if self.text_pane.cursor.line() + 1 >= self.text_pane.lines_above_scroll() + self.text_pane.number_of_visible_lines(self.widget_data.size.height) {
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
        self.text_pane = state;
    }

}


app!(TextWidget);

