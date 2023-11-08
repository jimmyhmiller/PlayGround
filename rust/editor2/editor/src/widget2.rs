use std::{error::Error, str::from_utf8, any::Any, cell::RefCell, path::PathBuf, fs::File, io::Read};

use framework::KeyboardInput;
use futures::channel::mpsc::Sender;
use serde::{Serializer, Deserializer, Deserialize, Serialize};
use skia_safe::{Canvas, Rect, Font, FontStyle, Typeface, Path, RRect, Point, Data};

use crate::{widget::{Size, Position, TextOptions}, color::Color, util::{encode_base64, decode_base64}, wasm_messenger::{Message, Payload, DrawCommands, OutMessage, OutPayload}};

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


fn wrap_payload(payload: Payload) -> Message {
    // TODO: Message id
    Message {
        message_id: 0,
        wasm_id: 0,
        payload,
    }
}

impl Default for Box<dyn Widget> {
    fn default() -> Self {
        Box::new(())
    }
}
pub struct WasmWidget {
    pub sender: Sender<Message>,
    pub draw_commands: Vec<DrawCommands>,
    pub receiver: futures::channel::mpsc::Receiver<OutMessage>,
    pub meta: WidgetMeta,
    // TODO:
    // Maybe we make a "mark dirty" sender
    // That way each widget can decide it is dirty
    // and needs to be drawn just by sending a message
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

        canvas.save();
        canvas.translate((self.position().x, self.position().y));
        canvas.scale((self.scale(), self.scale()));

        let mut current_width = 0.0;
        let mut current_height = 0.0;
        let mut current_height_stack = vec![];

        let mut paint = skia_safe::Paint::default();
        for command in self.draw_commands.iter() {
            match command {
                DrawCommands::SetColor(r, g, b, a) => {
                    let color = Color::new(*r, *g, *b, *a);
                    paint.set_color(color.as_color4f().to_color());
                }
                DrawCommands::DrawRect(x, y, width, height) => {
                    canvas
                        .draw_rect(skia_safe::Rect::from_xywh(*x, *y, *width, *height), &paint);
                }
                DrawCommands::DrawString(str, x, y) => {
                    let mut paint = paint.clone();
                    paint.set_shader(None);
                    if current_height > bounds.height {
                        continue;
                    }
                    if current_height < 0.0 {
                        continue;
                    }
                    let font = Font::new(
                        Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
                        32.0,
                    );

                    // No good way right now to find bounds. Need to think about this properly
                    canvas.draw_str(str, (*x, *y), &font, &paint);
                }

                DrawCommands::ClipRect(x, y, width, height) => {
                    canvas.clip_rect(
                        skia_safe::Rect::from_xywh(*x, *y, *width, *height),
                        None,
                        None,
                    );
                }
                DrawCommands::DrawRRect(x, y, width, height, radius) => {
                    let rrect = skia_safe::RRect::new_rect_xy(
                        skia_safe::Rect::from_xywh(*x, *y, *width, *height),
                        *radius,
                        *radius,
                    );
                    canvas.draw_rrect(rrect, &paint);
                }
                DrawCommands::Translate(x, y) => {
                    current_height += *y;
                    current_width += *x;
                    canvas.translate((*x, *y));
                }
                DrawCommands::Save => {
                    canvas.save();
                    current_height_stack.push(current_height);
                }
                DrawCommands::Restore => {
                    canvas.restore();
                    current_height = current_height_stack.pop().unwrap();
                }
            }
        }
        canvas.translate((self.size().width, 0.0));
        canvas.restore();

        Ok(())
    }

    fn on_click(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnClick(Position { x, y })))?;
        Ok(())
    }

    fn on_key(&mut self, input: KeyboardInput) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnKey(crate::keyboard::KeyboardInput::from_framework((input)))))?;
        Ok(())
    }

    fn on_scroll(&mut self, x: f64, y: f64) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnScroll(x, y)))?;
        Ok(())
    }

    fn on_size_change(&mut self, width: f32, height: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnSizeChange(width, height)))?;
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.meta.position = Position { x, y };
        self.sender.try_send(wrap_payload(Payload::OnMove(x, y)))?;
        Ok(())
    }

    fn save(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::SaveState))?;
        Ok(())
    }
    
    fn reload(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::Reload))?;
        Ok(())
    }

    fn set_state(&mut self, state: String) -> std::result::Result<(), Box<dyn Error>> {
        let base64_decoded = decode_base64(&state.as_bytes().to_vec()).unwrap();
        let state = String::from_utf8(base64_decoded).unwrap();
        self.sender.try_send(wrap_payload(Payload::PartialState(Some(state))))?;
        Ok(())
    }

    fn start(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        // self.sender.try_send(wrap_payload(Payload::Start))?;
        Ok(())
    }

    fn on_mouse_up(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnMouseUp(Position { x, y })))?;
        Ok(())
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnMouseDown(Position { x, y })))?;
        Ok(())
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::OnMouseMove(Position { x, y }, x_diff, y_diff)))?;
        Ok(())
    }

    fn on_event(&mut self, kind: String, event: String) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::Event(kind, event)))?;
        Ok(())
    }

    fn on_process_message(&mut self, process_id: i32, message: String) -> std::result::Result<(), Box<dyn Error>> {
        self.sender.try_send(wrap_payload(Payload::ProcessMessage(process_id as usize, message)))?;
        Ok(())
    }

    fn update(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        // TODO: Change this
        self.sender.try_send(wrap_payload(Payload::Draw("draw".to_string())))?;
        self.sender.try_send(wrap_payload(Payload::Update))?;
        while let Ok(Some(message)) = self.receiver.try_next() {
            // Note: Right now if a message doesn't have a corresponding in-message
            // I am just setting the out message to id: 0.
            // if let Some(record) = self.pending_messages.get_mut(&message.wasm_id) {
            //     record.remove(&message.message_id);
            // } else {
            //     println!("No pending message for {}", message.wasm_id)
            // }

            // TODO: This just means we update everything every frame
            // Because the draw content might not have changed
            // but we still request it every frame. We should only request it
            // if things have actually changed
            // Or we should only consider it dirty if things changed.

            let mut should_mark_dirty = true;

            // println!("Got message: {:?}", message.payload);

            match message.payload {
                OutPayload::DrawCommands(commands) => {
                    // TODO: Is this a performance issue?
                    // I'm thinking not? It seems to actually work because
                    // we only do this every once in a while
                    // if self.draw_commands.get(&message.wasm_id) == Some(&commands) {
                    //     should_mark_dirty = false;
                    // }
                    // self.wasm_draw_commands.insert(message.wasm_id, commands);
                    self.draw_commands = commands;
                }
                OutPayload::Update(commands) => {
                    // let current_commands = self
                    //     .wasm_non_draw_commands
                    //     .entry(message.wasm_id)
                    //     .or_default();
                    // if current_commands.is_empty() {
                    //     should_mark_dirty = false;
                    // }
                    // current_commands.extend(commands);
                }
                OutPayload::Saved(saved) => {
                    // self.wasm_states.insert(message.wasm_id, saved);
                }
                OutPayload::ErrorPayload(error_message) => {
                    println!("Error: {}", error_message);
                }
                OutPayload::NeededValue(name, sender) => {
                    // If I don't have the value, what should I do?
                    // Should I save this message and re-enqueue or signal failure?
                    // if let Some(value) = values.get(&name) {
                    //     let serialized = serde_json::to_string(value).unwrap();
                    //     sender.send(serialized).unwrap();
                    // } else {
                    //     // println!("Can't find value {}", name);
                    // }
                    // should_mark_dirty = false;
                }
                OutPayload::Reloaded => {
                    // TODO: Don't need widget id
                    // from message, but probably do need
                    // to know what widget we are.
                    // let commands = self
                    //     .wasm_non_draw_commands
                    //     .entry(message.wasm_id)
                    //     .or_default();
                    // commands.push(Commands::Redraw(widget_id));
                }
                OutPayload::Complete => {
                    // should_mark_dirty = false;
                }
                OutPayload::Error(error) => {
                    println!("Error: {}", error);
                }
            }
        }
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

#[derive(Serialize, Deserialize, Clone)]
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

#[derive(Serialize, Deserialize, Clone)]
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

pub struct Text {
    pub text: String,
    pub text_options: TextOptions,
    pub meta: WidgetMeta,
}

impl Widget for Text{
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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

    fn draw(&mut self, canvas: &Canvas, _bounds: Size) -> Result<(), Box<dyn Error>> {
        canvas.save();
        canvas.scale((self.scale(), self.scale()));
        let font = Font::new(
            Typeface::new(
                self.text_options.font_family.clone(),
                self.text_options.font_weight.into(),
            )
            .unwrap(),
            self.text_options.size,
        );
        let paint = self.text_options.color.as_paint();
        canvas.draw_str(
            self.text.clone(),
            (self.position().x, self.position().y + self.size().height),
            &font,
            &paint,
        );
        canvas.restore();
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct Image {
    pub path: String,
    // I am not sure about having this local
    // One thing I should maybe consider is only have
    // images in memory if they are visible.
    // How to do that though? Do I have a lifecycle for widgets
    // no longer being visible?
    // If I handled this globally all of that might be easier.
    #[serde(skip)]
    pub cache: RefCell<Option<skia_safe::Image>>,
    pub meta: WidgetMeta,
}

impl Widget for Image {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
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

    fn draw(&mut self, canvas: &Canvas, _bounds: Size) -> Result<(), Box<dyn Error>> {
        canvas.save();
        canvas.scale((self.scale(), self.scale()));
        // I tried to abstract this out and ran into the issue of returning a ref.
        // Can't use a closure, could box, but seems unnecessary. Maybe this data belongs elsewhere?
        // I mean the interior mutability is gross anyway.
        let image = self.cache.borrow();
        if image.is_none() {
            // Need to drop because we just borrowed.
            drop(image);
            self.load_image();
        }
        let image = self.cache.borrow();
        let image = image.as_ref().unwrap();
        canvas.draw_image(image, self.position(), None);
        canvas.restore();
        Ok(())
    }
}

impl Image {

    fn load_image(&self) {
        // TODO: Get rid of clone
        let path = if self.path.starts_with("./") {
            let mut base = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            base.push(&self.path);
            base.to_str().unwrap().to_string()
        } else {
            self.path.clone()
        };
        let mut file = File::open(path).unwrap();
        let mut image_data = vec![];
        file.read_to_end(&mut image_data).unwrap();
        let image = skia_safe::Image::from_encoded(Data::new_copy(image_data.as_ref())).unwrap();
        self.cache.replace(Some(image));
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