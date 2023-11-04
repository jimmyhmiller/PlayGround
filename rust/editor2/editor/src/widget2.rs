use std::{error::Error, str::from_utf8, any::Any};

use framework::KeyboardInput;
use futures::channel::mpsc::Sender;
use serde::{Serializer, Deserializer, Deserialize, Serialize};
use skia_safe::{Canvas, Rect, Font, FontStyle, Typeface, Path, RRect, Point};

use crate::{widget::{Size, Position}, color::Color, util::{encode_base64, decode_base64}};

#[allow(unused)]
pub trait Widget {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn draw(&mut self, canvas: &Canvas, bounds: Size) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>>{
        Ok(())
    }
    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) -> Result<(), Box<dyn Error>>{
        Ok(())
    }
    fn on_key(&mut self, input: KeyboardInput) -> Result<(), Box<dyn Error>>{
        Ok(())
    }
    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>>{
        Ok(())
    }
    fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>>{
        Ok(())
    }
    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>>{
        Ok(())
    }
    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn set_state(&mut self, state: String) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_process_message(&mut self, _process_id: i32, _message: String) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn save(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn reload(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn update(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn position(&self) -> Position;
    fn scale(&self) -> f32;
    fn size(&self) -> Size;
}

#[allow(unused)]
pub fn widget_serialize<S>(value: &Box<dyn Widget>, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_str(todo!())
}

#[allow(unused)]
pub fn widget_deserialize<'de, D>(deserializer: D) -> Result<Box<dyn Widget>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    todo!()

}

#[allow(unused)]
impl Widget for () {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn position(&self) -> Position {
        todo!()
    }

    fn scale(&self) -> f32 {
        todo!()
    }

    fn size(&self) -> Size {
        todo!()
    }
}

#[derive(Debug, Clone)]
enum Event {
    Start,
    OnClick(Position),
    Draw,
    OnScroll(f64, f64),
    OnKey(KeyboardInput),
    Reload,
    SaveState,
    ProcessMessage(usize, String),
    Event(String, String),
    OnSizeChange(f32, f32),
    OnMouseMove(Position, f32, f32),
    SetState(Option<String>),
    OnMouseDown(Position),
    OnMouseUp(Position),
    Update,
    OnMove(f32, f32),
}

impl Default for Box<dyn Widget> {
    fn default() -> Self {
        Box::new(())
    }
}
struct WasmWidget {
    sender: Sender<Event>,
}

#[allow(unused)]
impl Widget for WasmWidget {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn draw(&mut self, canvas: &Canvas, bounds: Size) -> Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::Draw)?;
        Ok(())
    }

    fn on_click(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnClick(Position { x, y }))?;
        Ok(())
    }

    fn on_key(&mut self, input: KeyboardInput) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnKey(input))?;
        Ok(())
    }

    fn on_scroll(&mut self, x: f64, y: f64) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnScroll(x, y))?;
        Ok(())
    }

    fn on_size_change(&mut self, width: f32, height: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnSizeChange(width, height))?;
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnMove(x, y))?;
        Ok(())
    }

    fn save(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::SaveState)?;
        Ok(())
    }
    
    fn reload(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::Reload)?;
        Ok(())
    }

    fn set_state(&mut self, state: String) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::SetState(Some(state)))?;
        Ok(())
    }

    fn start(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::Start)?;
        Ok(())
    }

    fn on_mouse_up(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnMouseUp(Position { x, y }))?;
        Ok(())
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnMouseDown(Position { x, y }))?;
        Ok(())
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::OnMouseMove(Position { x, y }, x_diff, y_diff))?;
        Ok(())
    }

    fn on_event(&mut self, kind: String, event: String) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::Event(kind, event))?;
        Ok(())
    }

    fn on_process_message(&mut self, process_id: i32, message: String) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::ProcessMessage(process_id as usize, message))?;
        Ok(())
    }

    fn update(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(Event::Update)?;
        Ok(())
    }

    fn position(&self) -> Position {
        todo!()
    }

    fn scale(&self) -> f32 {
        todo!()
    }

    fn size(&self) -> Size {
        todo!()
    }
}

fn serialize_text<S>(x: &[u8], s: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    s.serialize_str(&encode_base64(from_utf8(x).unwrap()))
}

fn deserialize_text<'de, D>(d: D) -> Result<Vec<u8>, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(d)?;
    let bytes = s.into_bytes();
    if let Ok(s) = decode_base64(&bytes) {
        Ok(s)
    } else {
        // TODO: Fail?
        Ok(bytes)
    }
}

#[derive(Serialize, Deserialize)]
pub struct WidgetMeta {
    position: Position,
    scale: f32,
    size: Size,
}

impl WidgetMeta {
    pub fn new(position: Position, size: Size, scale: f32, ) -> Self {
        Self {
            position,
            scale,
            size,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct TextPane {
    #[serde(serialize_with = "serialize_text")]
    #[serde(deserialize_with = "deserialize_text")]
    contents: Vec<u8>,
    pub line_height: f32,
    pub offset: Position,
    pub meta: WidgetMeta,
}

#[allow(unused)]
impl TextPane {
    pub fn new(contents: Vec<u8>, line_height: f32, meta: WidgetMeta) -> Self {
        Self {
            contents,
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
            meta,
        }
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

    pub fn visible_lines(&self, height: f32) -> impl std::iter::Iterator<Item = &str> + '_ {
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

    pub fn fractional_line_offset(&self) -> f32 {
        self.offset.y % self.line_height
    }

    pub fn set_text(&mut self, output: &str) {
        self.contents = output.into();
    }
}


#[allow(unused)]
impl Widget for TextPane {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn draw(&mut self, canvas: &Canvas, bounds: Size) -> Result<(), Box<dyn Error>> {
        canvas.save();
        let text_pane = &self;
        let foreground = Color::parse_hex("#dc9941");
        let background = Color::parse_hex("#353f38");

        let paint = background.as_paint();
        canvas.save();
        canvas.translate((self.position().x, self.position().y));
        canvas.clip_rect(Rect::from_wh(bounds.width, bounds.height), None, false);
        canvas.scale((self.scale(), self.scale()));

        let bounding_rect = Rect::new(0.0, 0.0, bounds.width, bounds.height);

        let font = Font::new(
            Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
            32.0,
        );
        let mut path = Path::new();
        path.add_rect(bounding_rect.with_outset((30.0, 30.0)), None);

        let rrect = RRect::new_rect_xy(bounding_rect, 20.0, 20.0);
        canvas.draw_rrect(rrect, &paint);

        canvas.clip_rect(bounding_rect.with_inset((20, 20)), None, None);
        let fractional_offset = text_pane.fractional_line_offset();
        canvas.translate((
            30.0 - text_pane.offset.x,
            text_pane.line_height - fractional_offset + 10.0,
        ));

        for line in text_pane.visible_lines(self.size().height) {
            canvas.draw_str(line, Point::new(0.0, 0.0), &font, &foreground.as_paint());
            canvas.translate((0.0, text_pane.line_height));
        }

        canvas.restore();
        canvas.restore();
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.meta.position.x = x;
        self.meta.position.y = y;
        Ok(())
    }

    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        self.on_scroll(x, y, self.size().height);
        Ok(())
    }

    fn position(&self) -> Position {
        self.meta.position
    }

    fn scale(&self) -> f32 {
        self.meta.scale
    }

    fn size(&self) -> Size {
        self.meta.size
    }
}


// These are things widgets can do.
// But we also need to store their locations and stuff
// so that we know when they are clicked or scrolled or whatever.
// This setup separates the widget and the wasm instance undeneath
// We just have a sender that lets that wasminstance receive events.
// I think this is a reasonable path.
// But I am worried about taking on a refactor rather than getting things working
// The good thing is my refactor keeps all my editor functionality in tact
// This one nice side-effect of the way I've written all of this.
// I can play with the shell independent of the editor details
// The ultimate goal of this refactor is to make it easy for wasm
// modules to have multiple panes.