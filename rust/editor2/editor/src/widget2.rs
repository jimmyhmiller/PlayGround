use std::{error::Error, collections::HashSet};

use framework::{Position, KeyboardInput};
use futures::channel::mpsc::Sender;
use skia_safe::Canvas;

use crate::wasm_messenger::WasmMessenger;

pub trait Widget {
    fn start(&mut self) -> Result<(), Box<dyn Error>>;
    fn draw(&mut self) -> Result<(), Box<dyn Error>>;
    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>>;
    fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>>;
    fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>>;
    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) -> Result<(), Box<dyn Error>>;
    fn on_key(&mut self, input: KeyboardInput) -> Result<(), Box<dyn Error>>;
    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>>;
    fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>>;
    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>>;
    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>>;
    fn set_state(&mut self, state: String) -> Result<(), Box<dyn Error>>;
    fn on_process_message(&mut self, _process_id: i32, _message: String) -> Result<(), Box<dyn Error>>;
    fn save(&mut self) -> std::result::Result<(), Box<dyn Error>>;
    fn reload(&mut self) -> std::result::Result<(), Box<dyn Error>>;
    fn update(&mut self) -> std::result::Result<(), Box<dyn Error>>;
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

struct WasmWidget {
    sender: Sender<Event>,
}

impl Widget for WasmWidget {
    fn draw(&mut self) -> std::result::Result<(), Box<dyn Error>> {
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
}

struct WidgetWithMessenger<'a> {
    widget: crate::widget::Widget,
    wasm_messenger: &'a mut WasmMessenger,
    canvas: &'a Canvas
}

impl Widget for WidgetWithMessenger {
    fn start(&mut self) -> Result<(), Box<dyn Error>> {
        self.widget.init(self.wasm_messenger);
        Ok(())
    }

    fn draw(&mut self) -> Result<(), Box<dyn Error>> {
        self.widget.draw(self.canvas, self.wasm_messenger, self.widget.size);
        Ok(())
    }

    fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_click(&crate::widget::Position{ x, y }, self.wasm_messenger);
        Ok(())
    }

    fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_mouse_up(&crate::widget::Position{ x, y }, self.wasm_messenger);
        Ok(())
    }

    fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_mouse_down(&crate::widget::Position{ x, y }, self.wasm_messenger);
        Ok(())
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn on_key(&mut self, input: KeyboardInput) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_size_change(width, height, self.wasm_messenger);
        Ok(())
    }

    fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.widget.on_move(x, y, self.wasm_messenger);
        Ok(())
    }

    fn set_state(&mut self, state: String) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn on_process_message(&mut self, _process_id: i32, _message: String) -> Result<(), Box<dyn Error>> {
        todo!()
    }

    fn save(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        todo!()
    }

    fn reload(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        todo!()
    }

    fn update(&mut self) -> std::result::Result<(), Box<dyn Error>> {
        todo!()
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