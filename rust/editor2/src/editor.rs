
use crate::fps_counter::FpsCounter;


use skia_safe::{RRect};
use winit::event::{Event as WinitEvent, WindowEvent as WinitWindowEvent};


pub enum Event {
    Noop,
    MouseMove { x: f64, y: f64 },
    LeftMouseDown,
    LeftMouseUp,
    RightMouseDown,
    RightMouseUp,
    Scroll { x: f64, y: f64 },
}

impl Event {
    fn from_winit_event(event: &WinitEvent<'_, ()>) -> Option<Self> {
        match event {
            WinitEvent::WindowEvent { event, .. } => {
                use WinitWindowEvent::*;
                match event {
                    CloseRequested => Some(Event::Noop),
                    TouchpadPressure {device_id: _, pressure: _, stage: _} => Some(Event::Noop),
                    MouseWheel { delta, .. } => {
                        match delta {
                            winit::event::MouseScrollDelta::LineDelta(_, _) => panic!("What is line delta?"),
                            winit::event::MouseScrollDelta::PixelDelta(delta) => Some(Event::Scroll { x: delta.x, y: delta.y })
                        }
                        
                    }
                    MouseInput { state, button, .. } => {
                        use winit::event::MouseButton::*;
                        use winit::event::ElementState::*;
                        match (state, button) {
                            (Pressed, Left) => Some(Event::LeftMouseDown),
                            (Released, Left) => Some(Event::LeftMouseUp),
                            (Pressed, Right) => Some(Event::RightMouseDown),
                            (Released, Right) => Some(Event::RightMouseUp),
                            _ => None,
                        }

                    }
                    // TODO: Cursor moves happen even without moving. Probably need to deduplicate somewhere
                    CursorMoved { position, .. }  => Some(Event::MouseMove { x: position.x, y: position.y }),
                    _ => {
                        println!("Unhandled event: {:?}", event);
                        None
                    }
                }
            },
            _ => {
                None
            },
        }
    }
}

pub struct Position {
    pub x: f64,
    pub y: f64,
}

pub struct ViewPane {
    pub resetting: bool,
    pub position: Position,
}


pub struct Editor {
    pub events: Vec<Event>,
    pub fps_counter: FpsCounter,
    pub view_pane: ViewPane,
}


#[cfg(all(target_os = "macos"))]
use skia_safe::{Canvas, Color4f, Paint, Point, Rect};




fn parse_hex(hex: &str) -> Paint {

    let mut start = 0;
    if hex.starts_with("#") {
        start = 1;
    }

    let r = i64::from_str_radix(&hex[start..start+2], 16).unwrap() as f32;
    let g = i64::from_str_radix(&hex[start+2..start+4], 16).unwrap() as f32;
    let b = i64::from_str_radix(&hex[start+4..start+6], 16).unwrap() as f32;
    return Paint::new(Color4f::new(r / 255.0, g / 255.0, b / 255.0, 1.0), None);
}


impl<'a> Editor {

    pub fn update(&mut self) {

    }

    pub fn draw(&mut self, canvas: &mut Canvas) {  
        self.fps_counter.tick();
        use skia_safe::{FontStyle, Font, Typeface};
    
        let gray = parse_hex("#333333");
        
        canvas.clear(gray.color4f());
    
        let rect = Rect::from_xywh(30.0, 30.0, 1200.0, 400.0);
        let mut rrect = RRect::new_rect_xy(rect, 20.0, 20.0);

        let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
        let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);

        let purple = parse_hex("#1c041e");
        let yellow = parse_hex("#2c1805");
        let green = parse_hex("#011b1e");
        
        canvas.draw_rrect(rrect, &purple);

        rrect.offset((0.0, rect.height() + 30.0));
        canvas.draw_rrect(rrect, &yellow);


        rrect.offset((0.0, rect.height() + 30.0));
        canvas.draw_rrect(rrect, &green);

    
        canvas.draw_str(self.fps_counter.fps.to_string(), Point::new(0.0, 30.0), &font, white);


        
    }

    pub fn add_event(&mut self, event: &winit::event::Event<'_, ()>) {
        if let Some(event) = Event::from_winit_event(&event) {
            match event {
                Event::Noop => {},
                Event::MouseMove { x: _, y: _ } => {},
                Event::LeftMouseDown => {},
                Event::LeftMouseUp => {},
                Event::RightMouseDown => {},
                Event::RightMouseUp => {},
                Event::Scroll { x: _, y: _ } => {
                
                }
            }
            self.events.push(event);
        }
    }
}


