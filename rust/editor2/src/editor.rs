
use std::path::PathBuf;

use crate::{fps_counter::FpsCounter, widget::{WidgetId, Position, WidgetStore, Widget, Size, WidgetData, ImageData, TextPane, TextOptions, FontWeight, Color, Process}};

use ron::ser::PrettyConfig;

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
    HoveredFile { path: PathBuf, x: f32, y: f32 },
    DroppedFile { path: PathBuf, x: f32, y: f32 },
    HoveredFileCancelled,
    ClickedWidget { widget_id: WidgetId },
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
            Event::HoveredFile { x, y, .. } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
            Event::DroppedFile { x, y, .. } => {
                *x = mouse_pos.x;
                *y = mouse_pos.y;
            }
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
                    HoveredFile(path) => Some(Event::HoveredFile { path: path.to_path_buf(), x: -0.0, y: -0.0 }),
                    DroppedFile(path) => Some(Event::DroppedFile { path: path.to_path_buf(), x: -0.0, y: -0.0 }),
                    HoveredFileCancelled => Some(Event::HoveredFileCancelled),
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



struct Scene {
    widgets: Vec<WidgetId>,
}

struct Context {
    mouse_position: Position,
    left_mouse_down: bool,
    right_mouse_down: bool,
}




pub struct Editor {
    events: Events,
    pub fps_counter: FpsCounter,
    scenes: Vec<Scene>,
    current_scene: usize,
    context: Context,
    scene_selector: Vec<WidgetId>,
    widget_store: WidgetStore,
}


struct Events {
    events: Vec<Event>,
    frame_start_index: usize,
    frame_end_index: usize, // exclusive
}


// It might be unnecessary to have start and end.
// But I like the explicitness for now.
impl Events {
    fn new() -> Events {
        Events {
            events: Vec::new(),
            frame_start_index: 0,
            frame_end_index: 0,
        }
    }

    fn push(&mut self, event: Event) {
        self.events.push(event);
    }

    fn events_for_frame(&self) -> &[Event] {
        &self.events[self.frame_start_index..self.frame_end_index]
    }

    fn next_frame(&mut self) {
        self.frame_start_index = self.frame_end_index;
    }

    fn end_frame(&mut self) {
        self.frame_end_index = self.events.len();
    }
}




#[cfg(all(target_os = "macos"))]
use skia_safe::{Canvas, Color4f, Paint, Point, Rect};






impl<'a> Editor {

    pub fn set_mouse_position(&mut self, x: f32, y: f32) {
        self.context.mouse_position = Position { x, y };
    }

    fn get_widget_by_id(&self, id: WidgetId) -> Option<&Widget> {
        // This means I never gc widgets
        // But I could of course free them and keep
        // a free list around.
        // or use a map. Or do a linear search.
        // But fine for now
        self.widget_store.get(id)
    }


    pub fn end_frame(&mut self) {
        self.events.end_frame();
    }

    pub fn next_frame(&mut self) {
        self.events.next_frame();
    }


    // TODO: Figure out a better setup for this
    // part of the problem is that scenes are not first class widgets
    // I need to figure out that hierarchy.
    // It will be useful for something like slideshow software
    pub fn setup(&mut self) {

        let id = self.widget_store.add_widget(Widget {
            id: 0,
            position: Position { x: 200.0, y: 200.0 },
            size: Size { width: 200.0, height: 100.0 },
            data: WidgetData::Noop,
        });

        let image_id = self.widget_store.add_widget(Widget {
            id: 0,
            position: Position { x: 400.0, y: 400.0 },
            size: Size { width: 200.0, height: 100.0 },
            data: WidgetData::Image {
                data: ImageData::new("/Users/jimmyhmiller/Downloads/Jimmy’s new iPad/files/20479ab6a77feae4060e5ea19beee525_original.png".to_string()),
            },
        });

        self.scenes.push(Scene {
            widgets: vec![id, image_id],
        });

        let id = self.widget_store.add_widget(Widget {
            id: 0,
            position: Position { x: 300.0, y: 300.0 },
            size: Size { width: 100.0, height: 100.0 },
            data: WidgetData::Noop,
        });

        let id_circle = self.widget_store.add_widget(Widget {
            id: 0,
            position: Position { x: 500.0, y: 500.0 },
            size: Size { width: 100.0, height: 100.0 },
            data: WidgetData::Circle { radius: 10.0, color: Color::parse_hex("#ff0000") },
        });



        let text_pane_id = self.widget_store.add_widget(Widget { 
            id: 0,
            position: Position { x: 500.0, y: 600.0 },
            size: Size { width: 500.0, height: 500.0 },
            data: WidgetData::TextPane {
                text_pane: TextPane::new("".as_bytes().to_vec(), 40.0)
            },
        });

        let text_id = self.widget_store.add_widget(Widget { 
            id: 0,
            position: Position { x: 600.0, y: 100.0 },
            size: Size { width: 1000.0, height: 1000.0 },
            data: WidgetData::Text {
                text: "Lith".to_string(),
                text_options: TextOptions {
                    size: 120.0,
                    color: Color::parse_hex("#ffffff00"),
                    font_weight: FontWeight::Bold,
                    font_family: "Ubuntu Mono".to_string(),
                },
            },
        });



        let id_compound = self.widget_store.add_widget(Widget {
            id: 0,
            position: Position { x: 500.0, y: 500.0 },
            size: Size { width: 100.0, height: 100.0 },
            data: WidgetData::Compound { children: vec![id, id_circle, text_id] },
        });

        // Should I have a custom serialization if we reference other widgets?
        // Right now compound just points to id. A little weird for external systems
        // That is general is something we have to figure out
        let compound_widget = self.widget_store.get(id_compound).unwrap();
        let compound_ron = ron::ser::to_string_pretty(compound_widget, PrettyConfig::default().struct_names(true)).unwrap();

        let text_widget = self.widget_store.get_mut(text_pane_id).unwrap();
        match text_widget.data {
            WidgetData::TextPane { ref mut text_pane } => {
                // let contents = text_pane.contents.clone();
                // let str_contents = from_utf8(&contents).unwrap();
                // let new_contents = str_contents.replace(MY_STRING, &compound_ron);
                text_pane.set_contents(compound_ron.as_bytes().to_vec());
            }
            _ => {
                println!("Not a text pane");
            }
        };

        self.scenes.push(Scene {
            widgets: vec![text_pane_id, id_compound]
        });


        for i in 0..self.scenes.len() {
            let id = self.widget_store.add_widget(Widget {
                id: 0,
                position: Position { x: 20.0, y: i as f32 * 120.0 + 20.0 },
                size: Size { width: 100.0, height: 100.0 },
                data: WidgetData::Noop,
            });
            self.scene_selector.push(id);
            
        }

    }

    pub fn update(&mut self) {

        // Todo: Need to test that I am not missing any
        // events with my start and end
        

        for event in self.events.events_for_frame() {
            match event {

                Event::DroppedFile { path, x, y } => {
                    println!("HOVERED!");
                    let hovered = self.widget_store.add_widget(Widget {
                        id: 0,
                        position: Position { x: *x, y: *y },
                        size: Size { width: 200.0, height: 100.0 },
                        data: WidgetData::Process { process: Process::new(path.to_path_buf()) },
                    });
                    self.scenes[self.current_scene].widgets.push(hovered);
                }

                Event::HoveredFileCancelled => {
                    let mut found = None;
                    for widget in self.widget_store.iter() {
                        if let WidgetData::HoverFile{..} = widget.data {
                            found = Some(widget.id);
                            break;
                        }
                    }

                    if let Some(found) = found {
                        self.widget_store.remove(found);
                    }
                }

                Event::Scroll { x, y } => {
                    let mouse = self.context.mouse_position;
                    for widget in self.widget_store.iter_mut() {
                        if widget.mouse_over(&mouse) {
                            match &mut widget.data {
                                WidgetData::TextPane { text_pane } => {
                                    text_pane.scroll(*x, *y, widget.size.height);
                                },
                                _ => {}
                            }
                        }
                    }
                }
                _ => {}
            }
        }


        for widget in self.widget_store.iter_mut() {
            match &widget.data {
                WidgetData::HoverFile { path: _ } => {
                    widget.position.x = self.context.mouse_position.x;
                    widget.position.y = self.context.mouse_position.y;
                }
                _ => {}
            }
        }
    }

    pub fn new() -> Self {
        Self {
            events: Events::new(),
            fps_counter: FpsCounter::new(),
            scenes: vec![],
            current_scene: 0,
            context: Context {
                mouse_position: Position { x: 0.0, y: 0.0 },
                left_mouse_down: false,
                right_mouse_down: false,
            },
            scene_selector: vec![],
            widget_store: WidgetStore::new()
        }
    }

    // let purple = parse_hex("#1c041e");
    // let yellow = parse_hex("#2c1805");
    // let green = parse_hex("#011b1e");
    

    pub fn draw(&mut self, canvas: &mut Canvas) {  
        
        self.fps_counter.tick();
        use skia_safe::{Size};

        let gray = Color::parse_hex("#333333");
        canvas.clear(gray.to_color4f());


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
        for widget_id in self.scene_selector.iter() {
            let widget = self.get_widget_by_id(*widget_id).unwrap();
            let rect = Rect::from_xywh(widget.position.x, widget.position.y, widget.size.width, widget.size.height);
            let rrect = RRect::new_rect_xy(rect, 20.0, 20.0);
            let purple = &Color::parse_hex("#1c041e").to_paint();
           
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
                canvas.draw_rrect(rrect, purple);
            } else {
                canvas.draw_rrect(rrect, &purple);
            }
        }

        let scene = &self.scenes[self.current_scene];
        for widget_id in scene.widgets.iter() {
            if let Some(widget) = self.get_widget_by_id(*widget_id) {
                widget.draw(canvas, &self.widget_store);
            }
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
            self.respond_to_event(event);
        }
    }

    fn respond_to_event(&mut self, mut event: Event) {
        event.patch_mouse_event(&self.context.mouse_position);
        match event {
            Event::ClickedWidget { widget_id } => {

                // Not going to be the plan going forward, but let's do this for now    
                for (i, scene_selector) in self.scene_selector.iter().enumerate() {
                    if *scene_selector == widget_id {
                        self.current_scene = i;
                        break;
                    }
                }
                println!("Clicked widget {}", widget_id);
                self.events.push(event);
            }
            Event::Noop => {},
            Event::MouseMove { x, y } => {
                // Not pushing the event because there are too many
                println!("Move");
                self.context.mouse_position = Position { x, y };
            },
            Event::LeftMouseDown {..} => {
                self.events.push(event);
                self.context.left_mouse_down = true;
                self.add_clicks();
            },
            Event::LeftMouseUp {..} => {
                self.events.push(event);
                self.context.left_mouse_down = false;
                // Probably not the right place.
                // Maybe need events on last cycle?
            },
            Event::RightMouseDown {..} => {
                self.events.push(event);
                self.context.right_mouse_down = true;
            },
            Event::RightMouseUp {..} => {
                self.events.push(event);
                self.context.right_mouse_down = false;
            },
            Event::Scroll { x:_, y:_ } => {
                self.events.push(event);
            }
            Event::HoveredFile { path: _, x: _, y: _ } => {
                self.events.push(event);
            }
            Event::DroppedFile { path: _, x: _, y: _ } => {
                self.events.push(event);
            }
            Event::HoveredFileCancelled => {
                self.events.push(event);
            }
        }
    }

    fn add_clicks(&mut self) -> () {
        let mut clicked = vec![];
        // I would need some sort of hierarchy here
        for widget in self.widget_store.iter() {
            if widget.mouse_over(&self.context.mouse_position) {
                clicked.push(widget.id);
            }
        }
        for id in clicked {
            self.respond_to_event(Event::ClickedWidget { widget_id: id });
        }
    }

}




// I need a way for processes to add widgets
// I don't want them to need to remove widgets by holding onto a reference or something
// It should work like react, always rendering the whole thing.
// At the same time, there are effects that processes might want to run and we need to distinguish between these.

// If a process gives me some output that are the widgets I should render
// how do I store them? If they clear that output, I should remove them
// But that implies that I have a way to identify them.

// I could instead each frame add the widgets of each process. 
// Then I would have a mostly blank list of widgets


// You might also want to have widgets that outlive the process.
// Widgets that stay only when the processes output is saved might not be desirable.
// Or is it? 
// The process itself can be gone, but the output can still be there.
// In fact, is there really a distinction? Is the widget not just the code that generates it?

// But it is possible that it would be better/faster
// to keep track of what widgets some output produced.
// I hold onto some ids. If the output is gone, I remove the widgets.
// If the output changes, I remove the widgets and add the new ones.
// I could then maybe do some sort of key based diffing algorithm.

// I also want the ability to start from data and make visualizations
// That data might be persisted or in  buffer
// Maybe I should start with some json and making visualizations?
// This is the biggest problem, the space is so big, I don't want to rabbit hole on something.

// Another thing I'm struggling with is what should be built in
// vs what should be extension based
// Maybe I should default to built-in and then move to extension afterwards?
// Probably will have more to show for it.



// I think for events, I should have a per frame list of events.
// I can do this while keeping the vec of events flat just by keeping an index
// Then update gets a list of events that happened on that frame
// and can respond to them.





// Example demos
// Build my own powerpoint
// Bind properties to each other
// Build a platformer, that uses widgets as the level editor
// YJS as an external addition
// Browser based stuff




// Ideas:
// 
// Make a process widget. It is a file, not text.
// We can get via drag and drop.
// On click runs it and makes a TextPane of the output.
// We can track when that changes, parse the RON it produces
// and add widgets
// We could also parse events this way to inject events into the system
// I like RON in general. But should consider some other format. Sad about toml
//
// Make my own react renderer that talks on a socket and produces widgets