use std::str::from_utf8;

use framework::{App, Canvas, Color, Rect, KeyState};
use serde::{Serialize, Deserialize, Serializer, Deserializer};
mod framework;

#[derive(Copy, Clone, Serialize, Deserialize)]
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

#[derive(Serialize, Deserialize, Clone)]
pub struct TextPane {
    #[serde(serialize_with = "serialize_text")]
    #[serde(deserialize_with = "deserialize_text")]
    contents: Vec<u8>,
    line_height: f32,
    offset: Position,
}

impl TextPane {
    pub fn new(_contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            contents: "asdfa\nasfadf\nasdfa\nasfadf\nasdfa\nasfadf\nasdfa\nasfadf\nasdfa\nasfadf\nasdfa\nasfadf\n".bytes().collect(),
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
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
        ((height - 40.0) / self.line_height).ceil() as usize
    }

    // TODO: obviously need to not compute this everytime.
    fn get_lines(&self) -> impl std::iter::Iterator<Item = &str> + '_ {
        let text = std::str::from_utf8(&self.contents).unwrap();
        let lines = text.split('\n');
        lines
    }

    fn number_of_lines(&self) -> usize {
        self.get_lines().count()
    }

    fn visible_lines(&self, height: f32) -> impl std::iter::Iterator<Item = &str> + '_ {
        self.get_lines()
            .skip(self.lines_above_scroll())
            .take(self.number_of_visible_lines(height))
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

        if self.offset.y > scroll_with_last_line_visible + 20.0 {
            self.offset.y = scroll_with_last_line_visible + 20.0;
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
                size: Size { width: 300.0, height: 300.0 },
            },
            edit_position: 0,
        }
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();


        let foreground = Color::parse_hex("#62b4a6");
        let background = Color::parse_hex("#530922");

        let bounding_rect = Rect::new(0.0, 0.0, 300.0, 300.0);

        canvas.save();
        canvas.set_color(background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);
        // TODO: deal with fonts
        // let font = Font::new(
        //     Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
        //     32.0,
        // );

        canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));
        let fractional_offset = self.text_pane.fractional_line_offset();
        canvas.translate(
            self.widget_data.position.x + 30.0 - self.text_pane.offset.x,
            self.widget_data.position.y + self.text_pane.line_height - fractional_offset + 20.0,
        );

        canvas.set_color(foreground);
        for line in self.text_pane.visible_lines(self.widget_data.size.height) {
            canvas.draw_str(line, 0.0, 0.0);
            canvas.translate(0.0, self.text_pane.line_height);
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

    fn on_click(&mut self) {

    }

    fn on_key(&mut self, input: KeyboardInput) {
        if !matches!(input.state, KeyState::Pressed) {
            return;
        }
        if self.edit_position >= self.text_pane.contents.len() {
            self.edit_position = 0;
        }
        if let Some(char) = input.to_char() {
            self.text_pane.contents[self.edit_position] = char as u8;
            self.edit_position += 1;
        }
        if self.edit_position >= self.text_pane.contents.len() {
            self.edit_position = 0;
        }
        while self.text_pane.contents[self.edit_position] == b'\n' {
            self.edit_position += 1;
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
        self.text_pane = TextPane::new(vec![], 30.0);
    }



}


app!(TextWidget);

