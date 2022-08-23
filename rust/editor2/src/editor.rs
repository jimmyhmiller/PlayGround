
use std::{fs::File, io::{Read}, cell::{RefCell}};

use crate::fps_counter::FpsCounter;


use skia_safe::{RRect, Font, Typeface, FontStyle, PaintStyle, Image, Data};
use winit::event::{Event as WinitEvent, WindowEvent as WinitWindowEvent};


const MY_STRING : &str =
"asdfjknsdva
sdvasdklnv'asdm
asdmv
asdkg;klasdjnvjlkasdnvkljansdkljcnalksjdnvjklasndvljkansdjlkvnasldjknvaljksdnvljkasdnljkvansdjklvnasdlkjvnalskjdnvakljsdnvljkasdn
vma
sd;lvma
sd;lvmas
d;lvma
sdlvma
sd;lvmas
dlvma
;dlsmv
asldmv
alsdmv
alsdmva;lsdmv
asl;dmv
alsdmv
;alsmd
valsdl;l;asldl;gal;sd;lfl;asl;dfl;asdlkfasldkfl;askd
fla;skd
f;laksd'lfka'sdl;ka'dls;kf'als;d
";



pub enum Event {
    Noop,
    MouseMove { x: f32, y: f32 },
    LeftMouseDown { x: f32, y: f32 },
    LeftMouseUp { x: f32, y: f32 },
    RightMouseDown { x: f32, y: f32 },
    RightMouseUp { x: f32, y: f32 },
    Scroll { x: f64, y: f64 },
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


#[derive(Copy, Clone)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

struct Size {
    width: f32,
    height: f32,
}

struct Widget {
    pub id: WidgetId,
    pub position: Position,
    pub size: Size,
    // Children might make sense
    // pub children: Vec<Widget>,
    pub data : WidgetData
}




// I could go the interface route here.
// I like enums. Will consider it later.
enum WidgetData {
    Noop,
    Circle {
        radius: f32,
        color: Color4f,
    },
    Compound {
        children: Vec<WidgetId>,
    },
    Image {
       data: ImageData
    },
    TextPane {
        text_pane: TextPane,
    },
    Text {
        text: String,
        text_options: TextOptions,
    }
}

struct TextOptions {
    font_family: String,
    font_style: FontStyle,
    size: f32,
    color: Color4f,
}


struct TextPane {
    contents: Vec<u8>,
    line_height: f32,
    offset: Position,
}


impl TextPane {
    fn new(contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            contents,
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
        }
    }

    fn lines_above_scroll(&self) -> usize {
        (self.offset.y / self.line_height).floor() as usize
    }

    fn number_of_visible_lines(&self, height: f32) -> usize {
        (height / self.line_height).ceil() as usize
    }

    // TODO: obviously need to not compute this everytime.
    fn get_lines(&self) -> impl std::iter::Iterator<Item=&str> + '_ {
        let text = std::str::from_utf8(&self.contents).unwrap();
        let lines = text.split('\n');
        return lines;
    }
    
    fn number_of_lines(&self) -> usize {
        self.get_lines().count()
    }
    
    fn visible_lines(&self, height: f32) -> impl std::iter::Iterator<Item=&str> + '_  {
        self.get_lines().skip(self.lines_above_scroll()).take(self.number_of_visible_lines(height))
    }

    fn scroll(&mut self, x: f64, y: f64, height: f32) {
        self.offset.x += x as f32;
        if self.offset.x < 0.0 {
            self.offset.x = 0.0;
        }
        // TODO: Handle x scrolling too far
        self.offset.y -= y as f32;
        if self.lines_above_scroll() + self.number_of_visible_lines(height) >= self.number_of_lines() {
            self.offset.y = (self.number_of_lines() - self.number_of_visible_lines(height)) as f32 * self.line_height;
        }
        if self.offset.y < 0.0 {
            self.offset.y = 0.0;
        }
        println!("offset: {}", self.offset.y);

    }

    fn fractional_line_offset(&self) -> f32 {
        self.offset.y % self.line_height
    }


}

struct ImageData {
    path: String,
    // I am not sure about having this local
    // One thing I should maybe consider is only have
    // images in memory if they are visible.
    // How to do that though? Do I have a lifecycle for widgets
    // no longer being visible?
    // If I handled this globally all of that might be easier.
    cache: RefCell<Option<Image>>,
}

impl ImageData {
    fn new(path: String) -> Self {
        Self {
            path,
            cache: RefCell::new(None),
        }
    }

    fn load_image(&self) {
        let mut file = File::open(&self.path).unwrap();
        let mut image_data = vec![];
        file.read_to_end(&mut image_data).unwrap();
        let image = Image::from_encoded(Data::new_copy(image_data.as_ref())).unwrap();
        self.cache.replace(Some(image));
    }

}


impl Widget {

    fn bounding_rect(&self) -> Rect {
        Rect::from_xywh(self.position.x, self.position.y, self.size.width, self.size.height)
    }

    fn draw(&self, canvas: &mut Canvas, widgets: &WidgetStore) {
        match &self.data {
            WidgetData::Noop => {
                
                let rect = self.bounding_rect();
                let rrect = RRect::new_rect_xy(rect, 20.0, 20.0);
                let purple = parse_hex("#1c041e");
                canvas.draw_rrect(rrect, &to_paint(purple));

                let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
                let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
                canvas.draw_str("noop", Point::new(self.position.x + 30.0, self.position.y + 40.0), &font, white); 
            }

            WidgetData::Circle { radius, color } => {
                let center = Point::new(self.position.x + radius, self.position.y + radius);
                canvas.draw_circle(center, *radius, &to_paint(*color));
            }
            
            WidgetData::Compound { children } => {
                for child in children.iter() {
                    // Need to set coords to be relative to the parent widget?
                    // Or maybe I need two notions of position
                    // Or maybe there should be a distinction between a compound widget
                    // and a container or a scene or something.
                    let child_widget = widgets.get(*child).unwrap();
                    child_widget.draw(canvas, widgets);
                }
            }
            WidgetData::Image { data } => {
                // I tried to abstract this out and ran into the issue of returning a ref.
                // Can't use a closure, could box, but seems unnecessary. Maybe this data belongs elsewhere?
                // I mean the interior mutability is gross anyway.
                let image = data.cache.borrow();
                if image.is_none() {
                    // Need to drop because we just borrowed.
                    drop(image);
                    data.load_image();
                }
                let image = data.cache.borrow();
                let image = image.as_ref().unwrap();
                canvas.draw_image(image, (self.position.x, self.position.y), None);
            }
            WidgetData::TextPane { text_pane } => {
                canvas.save();
                canvas.clip_rect(self.bounding_rect(), None, None);
                let purple = parse_hex("#1c041e");
                canvas.draw_rect(self.bounding_rect(), &to_paint(purple));
                let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 32.0);
                let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);


                canvas.clip_rect(self.bounding_rect().with_inset((20,20)), None, None);
                let fractional_offset = text_pane.fractional_line_offset();
                canvas.translate((self.position.x + 30.0 - text_pane.offset.x, self.position.y - fractional_offset));

                for line in text_pane.visible_lines(self.size.height) {
                    canvas.draw_str(line, Point::new(0.0, 0.0), &font, white);
                    canvas.translate((0.0, text_pane.line_height));
                }

                canvas.restore();
            }
            WidgetData::Text { text, text_options } => {
                let font = Font::new(Typeface::new(text_options.font_family.clone(), text_options.font_style).unwrap(), text_options.size);
                let color = &Paint::new(text_options.color, None);
                canvas.draw_str(text, (self.position.x, self.position.y), &font, color);
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



struct Scene {
    widgets: Vec<WidgetId>,
}

struct Context {
    mouse_position: Position,
    left_mouse_down: bool,
    right_mouse_down: bool,
}


type WidgetId = usize;

struct WidgetStore {
    widgets: Vec<Widget>,
    next_id: WidgetId,
}

impl WidgetStore {
    fn add_widget(&mut self, mut widget: Widget) -> WidgetId {
        let id = self.next_id;
        self.next_id += 1;
        widget.id = id;
        self.widgets.push(widget);
        id
    }

    fn get(&self, id: usize) -> Option<&Widget> {
        // Is it -1?
        self.widgets.get(id)
    }

    fn new() -> WidgetStore {
        WidgetStore {
            widgets: Vec::new(),
            next_id: 0,
        }
    }
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




fn parse_hex(hex: &str) -> Color4f {

    let mut start = 0;
    if hex.starts_with("#") {
        start = 1;
    }

    let r = i64::from_str_radix(&hex[start..start+2], 16).unwrap() as f32;
    let g = i64::from_str_radix(&hex[start+2..start+4], 16).unwrap() as f32;
    let b = i64::from_str_radix(&hex[start+4..start+6], 16).unwrap() as f32;
    return Color4f::new(r / 255.0, g / 255.0, b / 255.0, 1.0)
}

fn to_paint(color: Color4f) -> Paint {
    Paint::new(color, None)
}


impl<'a> Editor {

    fn get_widget_by_id(&self, id: WidgetId) -> Option<&Widget> {
        // This means I never gc widgets
        // But I could of course free them and keep
        // a free list around.
        // or use a map. Or do a linear search.
        // But fine for now
        self.widget_store.get(id)
    }

    fn add_widget(&mut self, widget: Widget) -> WidgetId {
        let id = self.widget_store.add_widget(widget);
        id
    }

    pub fn end_frame(&mut self) {
        self.events.end_frame();
    }

    pub fn next_frame(&mut self) {
        self.events.next_frame();
    }


    pub fn setup(&mut self) {

        let id = self.add_widget(Widget {
            id: 0,
            position: Position { x: 200.0, y: 200.0 },
            size: Size { width: 200.0, height: 100.0 },
            data: WidgetData::Noop,
        });

        let image_id = self.add_widget(Widget {
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

        let id = self.add_widget(Widget {
            id: 0,
            position: Position { x: 300.0, y: 300.0 },
            size: Size { width: 100.0, height: 100.0 },
            data: WidgetData::Noop,
        });

        let id_circle = self.add_widget(Widget {
            id: 0,
            position: Position { x: 500.0, y: 500.0 },
            size: Size { width: 100.0, height: 100.0 },
            data: WidgetData::Circle { radius: 10.0, color: parse_hex("#ff0000") },
        });

        let text_id = self.add_widget(Widget { 
            id: 0,
            position: Position { x: 500.0, y: 600.0 },
            size: Size { width: 500.0, height: 500.0 },
            data: WidgetData::TextPane {
                text_pane: TextPane::new(MY_STRING.as_bytes().to_vec(), 40.0)
            },
        });

        let id_compound = self.add_widget(Widget {
            id: 0,
            position: Position { x: 500.0, y: 500.0 },
            size: Size { width: 100.0, height: 100.0 },
            data: WidgetData::Compound { children: vec![id, id_circle] },
        });

        self.scenes.push(Scene {
            widgets: vec![text_id, id_compound]
        });


        for i in 0..self.scenes.len() {
            let id = self.add_widget(Widget {
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
                Event::Scroll { x, y } => {
                    let mouse = self.context.mouse_position;
                    for widget in self.widget_store.widgets.iter_mut() {
                        if widget.mouse_over(&mouse) {
                            match &mut widget.data {
                                WidgetData::TextPane { text_pane } => {
                                    text_pane.scroll(*x, *y, widget.size.height);
                                },
                                _ => {}
                            }
                        }
                    }
                    println!("Scroll: {}, {}", x, y);
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

        let gray = parse_hex("#333333");
        canvas.clear(gray);


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
            let purple = &to_paint(parse_hex("#1c041e"));
           
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
            let widget = self.get_widget_by_id(*widget_id).unwrap();
            widget.draw(canvas, &self.widget_store);
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
            Event::Scroll { x, y } => {
                println!("Scroll {} {}", x,y );
                self.events.push(event);
            }
        }
    }

    fn add_clicks(&mut self) -> () {
        let mut clicked = vec![];
        // I would need some sort of hierarchy here
        for widget in self.widget_store.widgets.iter() {
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

