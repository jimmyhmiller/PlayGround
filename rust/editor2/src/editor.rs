
use crate::fps_counter::FpsCounter;


use skia_safe::{RRect, Font, Typeface, FontStyle, PaintStyle};
use winit::event::{Event as WinitEvent, WindowEvent as WinitWindowEvent};


pub enum Event {
    Noop,
    MouseMove { x: f32, y: f32 },
    LeftMouseDown { x: f32, y: f32 },
    LeftMouseUp { x: f32, y: f32 },
    RightMouseDown { x: f32, y: f32 },
    RightMouseUp { x: f32, y: f32 },
    Scroll { x: f64, y: f64 },
    ClickedWidget { widget_id: usize },
}

// Need a global store of widgets
// Then anything else refers to widgets by their id
// The nice part about that is that widgets can appear in multiple places
// It might make some other parts more awkward.
// I can also then deal with events to a widget as data with the id
// So things can listen for different interactions.
// If there are a large number of widgets, this could be a bottle neck
// So I could keep the hierachy. Like I know if the widget is not
// on the current scene then I am not going to click on it.


impl Event {

    fn patch_mouse_event(&mut self, mouse_pos: &Position) {
        match self {
            Event::LeftMouseDown { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            },
            Event::LeftMouseUp { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            },
            Event::RightMouseDown { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            },
            Event::RightMouseUp { x, y } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            },
            _ => {},
        }
    }

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
                            // Silly hack for the fact that I don't know positions here.
                            (Pressed, Left) => Some(Event::LeftMouseDown { x: -0.0, y: -0.0 }),
                            (Released, Left) => Some(Event::LeftMouseUp { x: -0.0, y: -0.0 }),
                            (Pressed, Right) => Some(Event::RightMouseDown { x: -0.0, y: -0.0 }),
                            (Released, Right) => Some(Event::RightMouseUp { x: -0.0, y: -0.0 }),
                            _ => None,
                        }

                    }
                    CursorMoved { position, .. }  => Some(Event::MouseMove { x: position.x as f32, y: position.y as f32 }),
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

    fn mouse_over(&self, position: &Position) -> bool {
        let x = position.x;
        let y = position.y;
        let x_min = self.position.x;
        let x_max = self.position.x + self.size.width;
        let y_min = self.position.y;
        let y_max = self.position.y + self.size.height;
        x >= x_min && x <= x_max && y >= y_min && y <= y_max
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

struct Context {
    mouse_position: Position,
    left_mouse_down: bool,
    right_mouse_down: bool,
}


pub struct Editor {
    pub events: Vec<Event>,
    pub fps_counter: FpsCounter,
    scenes: Vec<Scene>,
    current_scene: usize,
    context: Context,
    scene_selector: Vec<Widget>,
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


        for (i, _scene) in self.scenes.iter().enumerate() {

            self.scene_selector.push(Widget {
                position: Position { x: 20.0, y: i as f32 * 120.0 + 20.0 },
                size: Size { width: 100.0, height: 100.0 },
                data: WidgetData::Noop,
            });
            
        }

    }

    pub fn update(&mut self) {
       
    }

    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            fps_counter: FpsCounter::new(),
            scenes: vec![],
            current_scene: 0,
            context: Context {
                mouse_position: Position { x: 0.0, y: 0.0 },
                left_mouse_down: false,
                right_mouse_down: false,
            },
            scene_selector: vec![],
        }
    }

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
        for widget in self.scene_selector.iter() {
            let rect = Rect::from_xywh(widget.position.x, widget.position.y, widget.size.width, widget.size.height);
            let rrect = RRect::new_rect_xy(rect, 20.0, 20.0);
            let purple = parse_hex("#1c041e");
           
            if widget.mouse_over(&self.context.mouse_position) {
                let mut outline = white.clone();
                // green_outline.set_stroke(Some(2.0));
                outline.set_style(PaintStyle::Stroke);
                outline.set_stroke_width(3.0);
                outline.set_anti_alias(true);
                canvas.draw_rrect(rrect, &outline);
                let rect = Rect::from_xywh(
                    widget.position.x + 5.0,
                    widget.position.y + 5.0,
                    widget.size.width - 10.0,
                    widget.size.height - 10.0
                );
                let rrect = RRect::new_rect_xy(rect, 15.0, 15.0);
                canvas.draw_rrect(rrect, &purple);
            } else {
                canvas.draw_rrect(rrect, &purple);
            }
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
        if let Some(mut event) = Event::from_winit_event(&event) {
            event.patch_mouse_event(&self.context.mouse_position);
            match event {
                // Should this even happen?
                Event::ClickedWidget { widget_id: _ } => {
                    println!("Clicked widget");
                }
                Event::Noop => {},
                Event::MouseMove { x, y } => {
                    // Not pushing the event because there are too many
                    self.context.mouse_position = Position { x, y };
                },
                Event::LeftMouseDown {..} => {
                    self.events.push(event);
                    self.context.left_mouse_down = true;
                },
                Event::LeftMouseUp {..} => {
                    self.events.push(event);
                    self.context.left_mouse_down = false;
                    // Probably not the right place.
                    // Maybe need events on last cycle?

                    self.add_clicks();
                },
                Event::RightMouseDown {..} => {
                    self.events.push(event);
                    self.context.right_mouse_down = true;
                },
                Event::RightMouseUp {..} => {
                    self.events.push(event);
                    self.context.right_mouse_down = false;
                },
                Event::Scroll { x: _, y: _ } => {
                    self.events.push(event);
                }
            }
        }
    }

    fn add_clicks(&mut self) -> () {
      
    }
}


