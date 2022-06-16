
use crate::fps_counter::FpsCounter;


use skia_safe::{RRect, Font, Typeface, FontStyle};
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
    pub x: f32,
    pub y: f32,
}

struct Size {
    width: f32,
    height: f32,
}

// I could go the interface route here.
// I like enums. Will consider it later.
enum WidgetData {
    Noop
}


impl Widget {
    fn draw(&self, canvas: &mut Canvas) {
        match self.data {
            WidgetData::Noop => {

                let rect = Rect::from_xywh(self.position.x + 10.0, self.position.y + 10.0, self.size.width, self.size.height);
                let rrect = RRect::new_rect_xy(rect, 20.0, 20.0);
                let purple = parse_hex("#1c041e");
                canvas.draw_rrect(rrect, &purple);

                let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
                let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
                canvas.draw_str("noop", Point::new(self.position.x + 30.0, self.position.y + 40.0), &font, white);
                
            }
        }
    }
}

struct Widget {
    pub position: Position,
    pub size: Size,
    // Children might make sense
    // pub children: Vec<Widget>,
    pub data : WidgetData
}


struct Scene {
    widgets: Vec<Widget>,
}


pub struct Editor {
    pub events: Vec<Event>,
    pub fps_counter: FpsCounter,
    scenes: Vec<Scene>,
    current_scene: usize,
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


    pub fn setup(&mut self) {
        self.scenes.push(Scene {
            widgets: vec![
                Widget {
                    position: Position { x: 200.0, y: 200.0 },
                    size: Size { width: 200.0, height: 100.0 },
                    data: WidgetData::Noop,
                }
            ]
        });

        self.scenes.push(Scene {
            widgets: vec![
                Widget {
                    position: Position { x: 0.0, y: 0.0 },
                    size: Size { width: 100.0, height: 100.0 },
                    data: WidgetData::Noop,
                }
            ]
        });
    }

    pub fn update(&mut self) {
       
    }

    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            fps_counter: FpsCounter::new(),
            scenes: vec![],
            current_scene: 0,
        }
    }

            // let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);

        // let purple = parse_hex("#1c041e");
        // let yellow = parse_hex("#2c1805");
        // let green = parse_hex("#011b1e");
        

    pub fn draw(&mut self, canvas: &mut Canvas) {  
        
        self.fps_counter.tick();
        use skia_safe::{Size};

        let gray = parse_hex("#333333");
        canvas.clear(gray.color4f());


        let canvas_size = Size::from(canvas.base_layer_size());

        let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
        let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
        canvas.draw_str(self.fps_counter.fps.to_string(), Point::new(canvas_size.width - 60.0, 30.0), &font, white);
        

        // Need to think in general about clicks
        // So I probably need to represent these as data
        // Do I want to have something at the top level tell
        // a widget if the mouse is over it?
        // Or do I want the widget to look at the mouse position?
        // Maybe allow for both?
        for (i, _scene) in self.scenes.iter().enumerate() {
            let rect = Rect::from_xywh(20.0, i as f32 * 120.0 + 20.0, 100.0, 100.0);
            let rrect = RRect::new_rect_xy(rect, 20.0, 20.0);
            let purple = parse_hex("#1c041e");
            let green = parse_hex("#011b1e");
            canvas.draw_rrect(rrect, if i == self.current_scene { &green } else { &purple });
        }

        let scene = &self.scenes[self.current_scene];
        for widget in scene.widgets.iter() {
            widget.draw(canvas);
        }




    
        // let rect = Rect::from_xywh(30.0, 30.0, 1200.0, 400.0);
        // let mut rrect = RRect::new_rect_xy(rect, 20.0, 20.0);

        // let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);

        // canvas.draw_rrect(rrect, &purple);

        // rrect.offset((0.0, rect.height() + 30.0));
        // canvas.draw_rrect(rrect, &yellow);


        // rrect.offset((0.0, rect.height() + 30.0));
        // canvas.draw_rrect(rrect, &green);



        
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


