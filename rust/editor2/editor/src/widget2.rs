use std::{
    any::Any,
    cell::RefCell,
    collections::HashMap,
    error::Error,
    fs::File,
    io::Read,
    ops::{Deref, DerefMut},
    path::PathBuf,
    time::Instant,
};

use cacao::input;
use framework::{KeyboardInput, Position, Size, Value, WidgetMeta};
use futures::channel::{mpsc::Sender, oneshot};
use itertools::Itertools;
use rand::Rng;
use serde::{Deserialize, Serialize};
use skia_safe::{
    font_style::{Slant, Weight, Width},
    Canvas, Data, Font, FontMgr, FontStyle, Path, Point, RRect, Rect, Typeface, Paint, Surface, surfaces::raster_n32_premul, canvas::SrcRectConstraint,
};

use crate::{
    color::Color,
    event::Event,
    wasm_messenger::{Commands, DrawCommands, Message, OutMessage, OutPayload, Payload, SaveState}, keyboard::{KeyState, KeyCode},
};

#[allow(unused)]
#[typetag::serde(tag = "type")]
pub trait Widget {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;

    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_mouse_move(
        &mut self,
        x: f32,
        y: f32,
        x_diff: f32,
        y_diff: f32,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_key(&mut self, input: KeyboardInput) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn set_state(&mut self, state: String) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn on_process_message(
        &mut self,
        _process_id: i32,
        _message: String,
    ) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
    fn update(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn get_state(&self) -> String;

    fn position(&self) -> Position;
    fn scale(&self) -> f32;
    fn set_scale(&mut self, scale: f32);
    fn size(&self) -> Size;
    fn id(&self) -> usize;
    fn parent_id(&self) -> Option<usize>;
    fn set_parent_id(&mut self, id: Option<usize>);
    fn set_id(&mut self, id: usize);

    fn dirty(&self) -> bool {
        true
    }

    fn mark_dirty(&mut self, reason: &str) {}

    fn reset_dirty(&mut self) {}

    fn as_wasm_widget(&self) -> Option<&WasmWidget> {
        self.as_any().downcast_ref::<WasmWidget>()
    }

    fn as_wasm_widget_mut(&mut self) -> Option<&mut WasmWidget> {
        self.as_any_mut().downcast_mut::<WasmWidget>()
    }
}

#[allow(unused)]
#[typetag::serde(name = "Unit")]
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

    fn set_scale(&mut self, scale: f32) {
        todo!()
    }

    fn size(&self) -> Size {
        todo!()
    }

    fn get_state(&self) -> String {
        todo!()
    }
    fn id(&self) -> usize {
        todo!()
    }

    fn set_id(&mut self, id: usize) {
        todo!()
    }
    fn parent_id(&self) -> Option<usize> {
        todo!()
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        todo!()
    }
}

impl Default for Box<dyn Widget> {
    fn default() -> Self {
        Box::new(())
    }
}
#[inline]
pub fn bool_true() -> bool {
    true
}
#[derive(Serialize, Deserialize)]
pub struct WasmWidget {
    #[serde(skip)]
    pub sender: Option<Sender<Message>>,
    #[serde(skip)]
    pub draw_commands: Vec<DrawCommands>,
    #[serde(skip)]
    pub receiver: Option<futures::channel::mpsc::Receiver<OutMessage>>,
    pub meta: WidgetMeta,
    pub save_state: SaveState,
    #[serde(skip)]
    pub wasm_non_draw_commands: Vec<Commands>,
    #[serde(skip)]
    pub external_sender: Option<std::sync::mpsc::Sender<Event>>,
    #[serde(default)]
    pub path: String,
    #[serde(default)]
    pub message_id: usize,
    #[serde(skip)]
    pub pending_messages: HashMap<usize, (Message, Instant)>,
    #[serde(default = "bool_true")]
    pub dirty: bool,
    pub external_id: Option<u32>,
    #[serde(skip)]
    pub value_senders: HashMap<String, oneshot::Sender<Value>>,
    #[serde(skip)]
    pub atlas: Option<skia_safe::Image>,
    #[serde(default)]
    pub offset: Position,
    #[serde(default)]
    pub size_offset: Size,
    // TODO:
    // Maybe we make a "mark dirty" sender
    // That way each widget can decide it is dirty
    // and needs to be drawn just by sending a message
}

pub fn draw_font_texture() -> Result<(skia_safe::Image, usize, usize), String> {
    let font = Font::new(
        Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
        32.0,
    );
    let mut text = String::new();
    for i in 33..127 {
        text.push(i as u8 as char);
    }
    let paint = Paint::new(Color::parse_hex("#ffffff").as_color4f(), None);
    let mut surface = raster_n32_premul(((text.len() * 90) as i32, 900 as i32)).unwrap();
    let canvas = surface.canvas();
    canvas.translate((0.0, 0.0));

    canvas.draw_str(text.clone(), (0.0, 30.0), &font, &paint);

    let image = surface.image_snapshot();
    let size = image.dimensions();
    let width = (size.width / text.len() as i32).try_into().unwrap();
    Ok((image, width, size.height.try_into().unwrap()))
}

#[allow(unused)]
#[typetag::serde]
impl Widget for WasmWidget {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn dirty(&self) -> bool {
        self.dirty
            || self.number_of_pending_requests() > 0
            || !self.wasm_non_draw_commands.is_empty()
            || self.draw_commands.is_empty()
    }

    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {

        let font = Font::new(
            Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
            32.0,
        );

        if self.atlas.is_none() {
            let (image, width, height) = draw_font_texture().unwrap();

            self.atlas = Some(image);
        }
        // TODO: Make not dumb
        
        let bounds = self.size();
        canvas.save();
        // canvas.translate((self.position().x, self.position().y));
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
                    canvas.draw_rect(skia_safe::Rect::from_xywh(*x, *y, *width, *height), &paint);
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

                    // The font atlas isn't quite right
                    // But also, it was no faster than skia
                    // Can I make it faster? Should I spend that time?
                    // Can I be smarter about what I render?
                    // Is text the thing slowing down code editor?
                    // Or maybe it is this whole draw_commands biz
                    // Maybe I need to profile
                    // self.draw_string(str, (*x, *y), &canvas);
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

    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::OnClick(Position { x, y }));
        self.send_message(message)?;
        Ok(())
    }

    fn on_key(&mut self, input: KeyboardInput) -> Result<(), Box<dyn Error>> {
        let input = crate::keyboard::KeyboardInput::from_framework(input);
        let message = self.wrap_payload(Payload::OnKey(input));
        
        // if input.state == KeyState::Pressed {
            // if input.modifiers.ctrl {
            //     match input.key_code {
            //         KeyCode::LeftArrow => {
            //             self.size_offset.width -= 1.0;
            //         }
            //         KeyCode::RightArrow => {
            //             self.size_offset.width += 1.0;
            //         }
            //         KeyCode::UpArrow => {
            //             self.size_offset.height -= 1.0;
            //         }
            //         KeyCode::DownArrow => {
            //             self.size_offset.height += 1.0;
            //         }
            //         _ => {}
            //     }
            // } else {
            //     match input.key_code {
            //         KeyCode::LeftArrow => {
            //             self.offset.x -= 1.0;
            //         }
            //         KeyCode::RightArrow => {
            //             self.offset.x += 1.0;
            //         }
            //         KeyCode::UpArrow => {
            //             self.offset.y -= 1.0;
            //         }
            //         KeyCode::DownArrow => {
            //             self.offset.y += 1.0;
            //         }
            //         _ => {}
            //     }
            // }
            
        // }
       
        // self.send_message(message)?;
        Ok(())
    }

    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::OnScroll(x, y));
        self.send_message(message)?;
        Ok(())
    }

    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>> {
        self.meta.size = Size { width, height };
        let message = self.wrap_payload(Payload::OnSizeChange(width, height));
        self.send_message(message)?;
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.meta.position = Position { x, y };
        let message = self.wrap_payload(Payload::OnMove(x, y));
        self.send_message(message)?;
        Ok(())
    }

    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::SaveState);
        self.send_message(message)?;
        Ok(())
    }

    fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::Reload);
        self.send_message(message)?;
        Ok(())
    }

    fn set_state(&mut self, state: String) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::PartialState(Some(state)));
        self.send_message(message)?;
        Ok(())
    }

    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        // self.sender.as_mut().unwrap().try_send(self.get_message(Payload::Start))?;
        Ok(())
    }

    fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::OnMouseUp(Position { x, y }));
        self.send_message(message)?;
        Ok(())
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::OnMouseDown(Position { x, y }));
        self.send_message(message)?;
        Ok(())
    }

    fn on_mouse_move(
        &mut self,
        x: f32,
        y: f32,
        x_diff: f32,
        y_diff: f32,
    ) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::OnMouseMove(Position { x, y }, x_diff, y_diff));
        self.send_message(message)?;
        Ok(())
    }

    fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::Event(kind, event));
        self.send_message(message)?;
        Ok(())
    }

    fn on_process_message(
        &mut self,
        process_id: i32,
        message: String,
    ) -> Result<(), Box<dyn Error>> {
        let message = self.wrap_payload(Payload::ProcessMessage(process_id as usize, message));
        self.send_message(message)?;
        Ok(())
    }

    fn update(&mut self) -> Result<(), Box<dyn Error>> {
        // TODO: Need to figure out how to deal with updates and dirty in
        // a way that doesn't suck

        let message = self.wrap_payload(Payload::GetCommands);
        self.send_message(message)?;

        let values = &mut HashMap::new();

        self.process_non_draw_commands(values);

        for (name, value) in values.drain() {
            self.external_sender
                .as_mut()
                .unwrap()
                .send(Event::ProvideValue(name, value));
        }

        while let Ok(Some(message)) = self.receiver.as_mut().unwrap().try_next() {
            // Note: Right now if a message doesn't have a corresponding in-message
            // I am just setting the out message to id: 0.

            if let Some((message, instant)) = self.pending_messages.remove(&message.message_id) {
                let duration = instant.elapsed();
                // println!("Message {:?} took {:?}", message.payload, duration);
            }

            // TODO: This just means we update everything every frame
            // Because the draw content might not have changed
            // but we still request it every frame. We should only request it
            // if things have actually changed
            // Or we should only consider it dirty if things changed.

            // println!("Got message: {:?}", message.payload);

            match message.payload {
                OutPayload::DrawCommands(commands) => {
                    // TODO: Is this a performance issue?
                    // I'm thinking not? It seems to actually work because
                    // we only do this every once in a while
                    if self.draw_commands != commands {
                        self.mark_dirty("draw_commands changed");
                        self.draw_commands = commands;
                    }
                }
                OutPayload::Update(commands) => {
                    if !commands.is_empty() {
                        self.mark_dirty("Got update");
                        self.wasm_non_draw_commands.extend(commands);
                    }
                }
                // OutPayload::MarkDirty => {
                //     self.mark_dirty("Mark dirty");
                // }
                OutPayload::Saved(saved) => {
                    self.save_state = saved;
                }
                OutPayload::ErrorPayload(error_message) => {
                    println!("Error: {}", error_message);
                }
                OutPayload::NeededValue(name, sender) => {
                    println!("Got needed value {}", name);
                    let id = self.id();
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::ValueNeeded(name.clone(), id));
                    self.value_senders.insert(name, sender);
                    // If I don't have the value, what should I do?
                    // Should I save this message and re-enqueue or signal failure?
                    // if let Some(value) = values.get(&name) {
                    //     let serialized = serde_json::to_string(value).unwrap();
                    //     sender.send(serialized).unwrap();
                    // } else {
                    //     // println!("Can't find value {}", name);
                    // }
                }
                OutPayload::Reloaded => {
                    // TODO: Don't need widget id
                    // from message, but probably do need
                    // to know what widget we are.
                    self.wasm_non_draw_commands
                        .push(Commands::Redraw(self.id()));
                    self.mark_dirty("Reload");
                }
                OutPayload::Complete => {
                    self.mark_dirty("complete");
                }
                OutPayload::Error(error) => {
                    println!("Error: {}", error);
                }
            }
        }
        if self.dirty() {
            let has_pending_draw = self
                .pending_messages
                .iter()
                .any(|(_, (m, _))| matches!(m.payload, Payload::RunDraw(_)));

            if !has_pending_draw {
                let message = self.wrap_payload(Payload::RunDraw("draw".to_string()));
                self.send_message(message)?;
            } else {
                println!("Already have pending draw for {}", self.id());
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

    fn set_scale(&mut self, scale: f32) {
        self.meta.scale = scale;
    }

    fn size(&self) -> Size {
        self.meta.size
    }

    fn get_state(&self) -> String {
        match &self.save_state {
            SaveState::Unsaved => "".to_string(),
            SaveState::Empty => "".to_string(),
            SaveState::Saved(s) => serde_json::to_string(s).unwrap(),
        }
    }

    fn id(&self) -> usize {
        self.meta.id
    }

    fn set_id(&mut self, id: usize) {
        self.meta.id = id;
    }

    fn parent_id(&self) -> Option<usize> {
        self.meta.parent_id
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        self.meta.parent_id = id;
    }

    fn mark_dirty(&mut self, reason: &str) {
        // println!("Marking dirty: {}", reason);
        self.dirty = true;
    }

    fn reset_dirty(&mut self) {
        self.dirty = false;
    }
}

impl WasmWidget {
    pub fn send_value(&mut self, name: String, value: Value) {
        if let Some(sender) = self.value_senders.remove(&name) {
            sender.send(value).unwrap();
        }
    }

    pub fn copy(&self, source: &Rect, canvas: &Canvas, (x, y): (f32, f32)) {
        let dst = Rect::from_xywh(0.0, 0.0, source.width(), source.height());
        let image = self.atlas.as_ref().unwrap();
        canvas.draw_image_rect(
            image,
            Some((source, SrcRectConstraint::Fast)),
            dst,
            &Paint::default(),
        );
    }


    pub fn move_right_one_char(&self, canvas: &Canvas) {
        canvas.translate((16.0, 0.0));
    }


    pub fn char_position_in_atlas(&self, c: char) -> Rect {
        let letter_width = 16.0;
        let letter_height = 30.0;
        // Rect::from_xywh((letter_width as i32 * (c as i32 - 33)) as f32, 0.0, letter_width as f32 + self.size_offset.width, letter_height as f32 + self.size_offset.height)
        Rect::from_xywh((letter_width as i32 * (c as i32 - 33)) as f32, 0.0, letter_width, letter_height)
        // Rect::from_xywh(0.0, 0.0, 300.0, 300.0)
    }

     pub fn draw_string(&self, text: &str, (x, y): (f32, f32), canvas: &Canvas) -> Result<(), String> {
        canvas.save();
        for char in text.chars() {
            self.move_right_one_char(canvas);
            self.copy(&self.char_position_in_atlas(char), canvas, (x, y));
        }
        canvas.restore();
        Ok(())
    }


    pub fn send_message(&mut self, message: Message) -> Result<(), Box<dyn Error>> {
        self.dirty = true;
        self.pending_messages
            .insert(message.message_id, (message.clone(), Instant::now()));
        self.sender.as_mut().unwrap().try_send(message)?;
        Ok(())
    }

    pub fn number_of_pending_requests(&self) -> usize {
        let non_draw_commands_count = self.wasm_non_draw_commands.len();
        let pending_message_count = self
            .pending_messages
            .iter()
            .filter(|(_, (message, _))| {
                !matches!(
                    message,
                    Message {
                        payload: Payload::RunDraw(_),
                        ..
                    }
                ) && !matches!(
                    message,
                    Message {
                        payload: Payload::GetCommands,
                        ..
                    }
                )
            })
            .collect::<Vec<_>>()
            .len();
        non_draw_commands_count + pending_message_count
    }

    pub fn pending_message_counts(&self) -> HashMap<String, usize> {
        let mut stats: Vec<&str> = vec![];
        for message in self.pending_messages.values() {
            stats.push(match message.0.payload {
                Payload::OnClick(_) => "OnClick",
                Payload::RunDraw(_) => "Draw",
                Payload::OnScroll(_, _) => "OnScroll",
                Payload::OnKey(_) => "OnKey",
                Payload::Reload => "Reload",
                Payload::SaveState => "SaveState",
                Payload::ProcessMessage(_, _) => "ProcessMessage",
                Payload::Event(_, _) => "Event",
                Payload::OnSizeChange(_, _) => "OnSizeChange",
                Payload::OnMouseMove(_, _, _) => "OnMouseMove",
                Payload::PartialState(_) => "PartialState",
                Payload::OnMouseDown(_) => "OnMouseDown",
                Payload::OnMouseUp(_) => "OnMouseUp",
                Payload::GetCommands => "Update",
                Payload::OnMove(_, _) => "OnMove",
                Payload::NewSender(_, _, _) => "NewSender",
            });
        }

        let counts = stats
            .iter()
            .counts()
            .iter()
            .map(|(k, v)| (k.to_string(), *v))
            .collect();
        counts
    }

    pub fn next_message_id(&mut self) -> usize {
        let current = self.message_id;
        self.message_id += 1;
        current
    }

    pub fn wrap_payload(&mut self, payload: Payload) -> Message {
        let message_id = self.next_message_id();
        Message {
            message_id,
            external_id: self.external_id,
            payload,
        }
    }

    pub fn process_non_draw_commands(&mut self, values: &mut HashMap<String, Value>) {
        let id = self.id();
        if !self.wasm_non_draw_commands.is_empty() {
            self.mark_dirty("non_draw_commands exist");
        }
        for command in self.wasm_non_draw_commands.iter() {
            match command {
                // This is probably routed incorrectly
                Commands::MarkDirty(widget_id) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::MarkDirty(*widget_id))
                        .unwrap();
                }
                Commands::StartProcess(process_id, process_command) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::StartProcess(
                            *process_id as usize,
                            // TODO: I probably actually want widget id?
                            id,
                            process_command.clone(),
                        ))
                        .unwrap();
                }
                Commands::SendProcessMessage(process_id, message) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::SendProcessMessage(
                            *process_id as usize,
                            message.clone(),
                        ))
                        .unwrap();
                }
                Commands::ReceiveLastProcessMessage(_) => println!("Unhandled"),
                Commands::ProvideValue(name, data) => {
                    // TODO: Get rid of clone here
                    values.insert(name.to_string(), data.clone());
                }
                Commands::Event(kind, event) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::Event(kind.clone(), event.clone()))
                        .unwrap();
                }
                Commands::Redraw(widget_id) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::Redraw(*widget_id))
                        .unwrap();
                }
                Commands::Subscribe(kind) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::Subscribe(
                            // TODO: I probably actually want widget id?
                            id,
                            kind.clone(),
                        ))
                        .unwrap();
                }
                Commands::Unsubscribe(kind) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::Unsubscribe(
                            // TODO: I probably actually want widget id?
                            id,
                            kind.clone(),
                        ))
                        .unwrap();
                }
                Commands::SetCursor(cursor) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::SetCursor(*cursor))
                        .unwrap();
                }
                Commands::CreateWidget(wasm_id, x, y, width, height, external_id) => {
                    self.external_sender
                        .as_mut()
                        .unwrap()
                        .send(Event::CreateWidget(
                            *wasm_id,
                            *x,
                            *y,
                            *width,
                            *height,
                            *external_id,
                        ))
                        .unwrap();
                }
            }
        }
        self.wasm_non_draw_commands.clear();
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TextPane {
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
#[typetag::serde]
impl Widget for TextPane {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        let bounds = self.size();
        canvas.save();
        let text_pane = &self;
        let foreground = Color::parse_hex("#dc9941");
        let background = Color::parse_hex("#353f38");

        let paint = background.as_paint();
        canvas.save();
        // canvas.translate((self.position().x, self.position().y));
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

    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>> {
        self.meta.size = Size { width, height };
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

    fn id(&self) -> usize {
        self.meta.id
    }

    fn set_id(&mut self, id: usize) {
        self.meta.id = id;
    }

    fn parent_id(&self) -> Option<usize> {
        self.meta.parent_id
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        self.meta.parent_id = id;
    }

    fn set_scale(&mut self, scale: f32) {
        self.meta.scale = scale;
    }

    fn get_state(&self) -> String {
        "".to_string()
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Light,
    Normal,
    Bold,
}

impl From<FontWeight> for FontStyle {
    fn from(val: FontWeight) -> Self {
        match val {
            FontWeight::Light => FontStyle::new(Weight::LIGHT, Width::NORMAL, Slant::Upright),
            FontWeight::Normal => FontStyle::normal(),
            FontWeight::Bold => FontStyle::bold(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct TextOptions {
    pub font_family: String,
    pub font_weight: FontWeight,
    pub size: f32,
    pub color: Color,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Text {
    pub text: String,
    pub text_options: TextOptions,
    pub meta: WidgetMeta,
}

#[typetag::serde]
impl Widget for Text {
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

    fn set_scale(&mut self, scale: f32) {
        self.meta.scale = scale;
    }

    fn size(&self) -> Size {
        self.meta.size
    }

    fn id(&self) -> usize {
        self.meta.id
    }

    fn set_id(&mut self, id: usize) {
        self.meta.id = id;
    }

    fn parent_id(&self) -> Option<usize> {
        self.meta.parent_id
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        self.meta.parent_id = id;
    }

    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
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
        canvas.draw_str(self.text.clone(), (0.0, self.size().height), &font, &paint);
        canvas.restore();
        Ok(())
    }
    fn get_state(&self) -> String {
        "".to_string()
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.meta.position.x = x;
        self.meta.position.y = y;
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
pub struct RandomText {
    pub text: Text,
    #[serde(default)]
    pub show_font: bool,
}

impl Deref for RandomText {
    type Target = Text;

    fn deref(&self) -> &Self::Target {
        &self.text
    }
}

impl DerefMut for RandomText {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.text
    }
}

#[typetag::serde]
impl Widget for RandomText {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_state(&self) -> String {
        self.text.get_state()
    }

    fn position(&self) -> Position {
        self.text.position()
    }

    fn scale(&self) -> f32 {
        self.text.scale()
    }

    fn set_scale(&mut self, scale: f32) {
        self.text.set_scale(scale)
    }

    fn size(&self) -> Size {
        self.text.size()
    }

    fn id(&self) -> usize {
        self.text.id()
    }

    fn set_id(&mut self, id: usize) {
        self.text.set_id(id)
    }

    fn parent_id(&self) -> Option<usize> {
        self.meta.parent_id
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        self.meta.parent_id = id;
    }

    // TODO: write font in text if mouse hover
    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.text_options = TextOptions {
            font_family: Self::random_font_name(),
            font_weight: FontWeight::Normal,
            size: 120.0,
            color: Color::parse_hex("#ffffff"),
        };
        self.text.on_click(x, y)
    }

    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        self.text_options = TextOptions {
            font_family: Self::random_font_name(),
            font_weight: FontWeight::Normal,
            size: 120.0,
            color: Color::parse_hex("#ffffff"),
        };
        self.text.start()
    }

    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        self.text.draw(canvas)?;
        if self.show_font {
            canvas.translate((0.0, 50.0));
            let font = Font::new(
                Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
                16.0,
            );
            canvas.draw_str(
                self.text_options.font_family.clone(),
                (0.0, 0.0),
                &font,
                &Color::parse_hex("#ffffff").as_paint(),
            );
        }
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.text.on_move(x, y)
    }

    fn on_key(&mut self, _input: KeyboardInput) -> Result<(), Box<dyn Error>> {
        self.show_font = !self.show_font;
        Ok(())
    }
}

impl RandomText {
    fn random_font_name() -> String {
        let font_mgr = FontMgr::default();
        let families_count = font_mgr.count_families();

        let random_font = rand::thread_rng().gen_range(0..families_count);
        let family_name = font_mgr.family_name(random_font);
        family_name.to_string()
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
    #[serde(default)]
    pub aspect_ratio: f32,
}

#[typetag::serde]
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

    fn set_scale(&mut self, scale: f32) {
        self.meta.scale = scale;
    }

    fn size(&self) -> Size {
        self.meta.size
    }

    fn id(&self) -> usize {
        self.meta.id
    }

    fn set_id(&mut self, id: usize) {
        self.meta.id = id;
    }

    fn parent_id(&self) -> Option<usize> {
        self.meta.parent_id
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        self.meta.parent_id = id;
    }

    fn get_state(&self) -> String {
        "".to_string()
    }

    fn dirty(&self) -> bool {
        true
    }

    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
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
        canvas.draw_image(image, (0.0, 0.0), None);
        canvas.restore();
        Ok(())
    }

    fn on_size_change(&mut self, width: f32, _height: f32) -> Result<(), Box<dyn Error>> {
        // TODO: This is wrong
        let width_diff = width / self.meta.size.width;
        self.meta.scale = width_diff * self.meta.scale;
        self.meta.size.width = width;
        self.meta.size.height = width / self.aspect_ratio;
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.meta.position.x = x;
        self.meta.position.y = y;
        Ok(())
    }
}

impl Image {
    fn load_image(&mut self) {
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
        self.meta.size = Size {
            width: image.bounds().width() as f32,
            height: image.bounds().height() as f32,
        };
        self.aspect_ratio = self.meta.size.width / self.meta.size.height;
        self.cache.replace(Some(image));
    }
}

#[derive(Serialize, Deserialize)]
pub struct Deleted {}

#[typetag::serde]
impl Widget for Deleted {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn position(&self) -> Position {
        Position { x: 0.0, y: 0.0 }
    }

    fn scale(&self) -> f32 {
        0.0
    }

    fn id(&self) -> usize {
        42
    }

    fn set_id(&mut self, _id: usize) {}

    fn parent_id(&self) -> Option<usize> {
        None
    }

    fn set_parent_id(&mut self, _id: Option<usize>) {
        
    }

    fn set_scale(&mut self, _scale: f32) {}

    fn size(&self) -> Size {
        Size {
            width: 0.0,
            height: 0.0,
        }
    }

    fn get_state(&self) -> String {
        "".to_string()
    }
}

#[derive(Serialize, Deserialize)]
pub struct Ephemeral {
    widget: Box<dyn Widget>,
}

impl Ephemeral {
    pub fn wrap(widget: Box<dyn Widget>) -> Ephemeral {
        Ephemeral { widget }
    }
}

impl Deref for Ephemeral {
    type Target = Box<dyn Widget>;

    fn deref(&self) -> &Self::Target {
        &self.widget
    }
}

impl DerefMut for Ephemeral {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.widget
    }
}

#[typetag::serde]
impl Widget for Ephemeral {
    fn as_any(&self) -> &dyn Any {
        self.widget.as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self.widget.as_any_mut()
    }

    fn get_state(&self) -> String {
        self.widget.get_state()
    }

    fn position(&self) -> Position {
        self.widget.position()
    }

    fn scale(&self) -> f32 {
        self.widget.scale()
    }

    fn set_scale(&mut self, scale: f32) {
        self.widget.set_scale(scale)
    }

    fn size(&self) -> Size {
        self.widget.size()
    }

    fn id(&self) -> usize {
        self.widget.id()
    }

    fn set_id(&mut self, id: usize) {
        self.widget.set_id(id)
    }

    fn parent_id(&self) -> Option<usize> {
        self.widget.parent_id()
    }

    fn set_parent_id(&mut self, id: Option<usize>) {
        self.widget.set_parent_id(id);
    }

    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        self.widget.start()
    }

    fn draw(&mut self, canvas: &Canvas) -> Result<(), Box<dyn Error>> {
        self.widget.draw(canvas)
    }

    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_click(x, y)
    }

    fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_mouse_up(x, y)
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_mouse_down(x, y)
    }

    fn on_mouse_move(
        &mut self,
        x: f32,
        y: f32,
        x_diff: f32,
        y_diff: f32,
    ) -> Result<(), Box<dyn Error>> {
        self.widget.on_mouse_move(x, y, x_diff, y_diff)
    }

    fn on_key(&mut self, input: KeyboardInput) -> Result<(), Box<dyn Error>> {
        self.widget.on_key(input)
    }

    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        self.widget.on_scroll(x, y)
    }

    fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>> {
        self.widget.on_event(kind, event)
    }

    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_size_change(width, height)
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_move(x, y)
    }

    fn set_state(&mut self, state: String) -> Result<(), Box<dyn Error>> {
        self.widget.set_state(state)
    }

    fn on_process_message(
        &mut self,
        _process_id: i32,
        _message: String,
    ) -> Result<(), Box<dyn Error>> {
        self.widget.on_process_message(_process_id, _message)
    }

    fn save(&mut self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }

    fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        self.widget.reload()
    }

    fn update(&mut self) -> Result<(), Box<dyn Error>> {
        self.widget.update()
    }

    fn dirty(&self) -> bool {
        self.widget.dirty()
    }

    fn mark_dirty(&mut self, reason: &str) {
        self.widget.mark_dirty(reason)
    }

    fn reset_dirty(&mut self) {
        self.widget.reset_dirty()
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

// TODO: I need to setup dirty widget stuff
// I'm not super happy with the whole meta thing.
