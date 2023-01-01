use std::{fs::File, process::ChildStdout, cell::RefCell, io::Read, path::PathBuf};

use nonblock::NonBlockingReader;
use serde::{Serialize, Deserialize};
use skia_safe::{Point, Paint, Color4f, FontStyle, font_style::{Weight, Width, Slant}, Image, Data, Rect, Canvas, RRect, Font, Typeface};

use crate::event::Event;


#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Into<Point> for Position {
    fn into(self) -> Point {
        Point { x: self.x, y: self.y }
    }
}

#[derive(Serialize, Deserialize)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}



pub type WidgetId = usize;

#[derive(Serialize, Deserialize)]
pub struct Widget {
    #[serde(skip)]
    pub id: WidgetId,
    pub position: Position,
    pub size: Size,
    pub on_click: Vec<Event>,
    // Children might make sense
    // pub children: Vec<Widget>,
    pub data : WidgetData
}

pub struct WidgetStore {
    widgets: Vec<Widget>,
    next_id: WidgetId,
}

impl WidgetStore {
    pub fn add_widget(&mut self, mut widget: Widget) -> WidgetId {
        let id = self.next_id;
        self.next_id += 1;
        widget.id = id;
        self.widgets.push(widget);
        id
    }

    pub fn get(&self, id: usize) -> Option<&Widget> {
        // Is it -1?
        self.widgets.get(id)
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Widget> {
        // Is it -1?
        self.widgets.get_mut(id)
    }

    pub fn new() -> WidgetStore {
        WidgetStore {
            widgets: Vec::new(),
            next_id: 0,
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Widget> {
        self.widgets.iter_mut()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Widget> {
        self.widgets.iter()
    }
}

#[derive(Serialize, Deserialize)]
pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {
    pub fn to_paint(&self) -> Paint {
        Paint::new(Color4f::new(self.r, self.g, self.b, self.a), None)
    }

    pub fn to_color4f(&self) -> Color4f {
        Color4f::new(self.r, self.g, self.b, self.a)
    }

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    pub fn parse_hex(hex: &str) -> Color {

        let mut start = 0;
        if hex.starts_with('#') {
            start = 1;
        }

        let r = i64::from_str_radix(&hex[start..start+2], 16).unwrap() as f32;
        let g = i64::from_str_radix(&hex[start+2..start+4], 16).unwrap() as f32;
        let b = i64::from_str_radix(&hex[start+4..start+6], 16).unwrap() as f32;
        Color::new(r / 255.0, g / 255.0, b / 255.0, 1.0)
    }
}



// I could go the interface route here.
// I like enums. Will consider it later.
#[derive(Serialize, Deserialize)]
pub enum WidgetData {
    Noop,
    Circle {
        radius: f32,
        color: Color,
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
    },
    Process {
        process: Process,
    },
    // I probably don't need this
    // But I think no with my mouse fix
    // I could do something cool with it?
    HoverFile {
        path: String,
    }
}

// TODO: watch for file changes
#[derive(Serialize, Deserialize)]
pub struct Process {
    file_path: PathBuf,
    #[serde(skip)]
    #[allow(dead_code)]
    file: Option<File>,
    #[serde(skip)]
    #[allow(dead_code)]
    stdout: Option<NonBlockingReader<ChildStdout>>
}

impl Process {
    pub fn new(file_path: PathBuf) -> Process {
        Process {
            file_path,
            file: None,
            stdout: None,
        }
    }
}



#[derive(Copy, Clone, Serialize, Deserialize)]
pub enum FontWeight {
    Light,
    Normal,
    Bold,
}

impl Into<FontStyle> for FontWeight {
    fn into(self) -> FontStyle {
        match self {
            FontWeight::Light => FontStyle::new(Weight::LIGHT, Width::NORMAL, Slant::Upright),
            FontWeight::Normal => FontStyle::normal(),
            FontWeight::Bold => FontStyle::bold(),
        }
    }
}


#[derive(Serialize, Deserialize)]
pub struct TextOptions {
    pub font_family: String,
    pub font_weight: FontWeight,
    pub size: f32,
    pub color: Color,
}


#[derive(Serialize, Deserialize)]
pub struct TextPane {
    contents: Vec<u8>,
    line_height: f32,
    offset: Position,
}


impl TextPane {
   pub fn new(contents: Vec<u8>, line_height: f32) -> Self {
        Self {
            contents,
            line_height,
            offset: Position { x: 0.0, y: 0.0 },
        }
    }

    pub fn set_contents(&mut self, contents: Vec<u8>) {
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
    fn get_lines(&self) -> impl std::iter::Iterator<Item=&str> + '_ {
        let text = std::str::from_utf8(&self.contents).unwrap();
        
        text.split('\n')
    }

    fn number_of_lines(&self) -> usize {
        self.get_lines().count()
    }

    fn visible_lines(&self, height: f32) -> impl std::iter::Iterator<Item=&str> + '_  {
        self.get_lines().skip(self.lines_above_scroll()).take(self.number_of_visible_lines(height))
    }

    pub fn scroll(&mut self, x: f64, y: f64, height: f32) {

        // this is all terribly wrong. I don't get the bounds correctly.
        // Need to deal with that.


        self.offset.x += x as f32;
        if self.offset.x < 0.0 {
            self.offset.x = 0.0;
        }
        // TODO: Handle x scrolling too far
        self.offset.y -= y as f32;

        let scroll_with_last_line_visible =
            self.number_of_lines().saturating_sub(self.number_of_visible_lines(height)) as f32 * self.line_height;


        // TODO: Deal with margin properly

        if self.offset.y > scroll_with_last_line_visible + 20.0 {
            self.offset.y = scroll_with_last_line_visible + 20.0 ;
        }


        if self.offset.y < 0.0 {
            self.offset.y = 0.0;
        }

    }

    fn fractional_line_offset(&self) -> f32 {
        self.offset.y % self.line_height
    }


}

#[derive(Serialize, Deserialize)]
pub struct ImageData {
    path: String,
    // I am not sure about having this local
    // One thing I should maybe consider is only have
    // images in memory if they are visible.
    // How to do that though? Do I have a lifecycle for widgets
    // no longer being visible?
    // If I handled this globally all of that might be easier.
    #[serde(skip)]
    cache: RefCell<Option<Image>>,
}

impl ImageData {
    pub fn new(path: String) -> Self {
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

    pub fn draw(&self, canvas: &mut Canvas, widgets: &WidgetStore) {
        match &self.data {
            WidgetData::Noop => {

                let rect = self.bounding_rect();
                let rrect = RRect::new_rect_xy(rect, 20.0, 20.0);
                let purple = Color::parse_hex("#1c041e");
                canvas.draw_rrect(rrect, &purple.to_paint());

                let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
                let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
                canvas.draw_str("noop", Point::new(self.position.x + 30.0, self.position.y + 40.0), &font, white);
            }

            WidgetData::Circle { radius, color } => {
                let center = Point::new(self.position.x + radius, self.position.y + radius);
                canvas.draw_circle(center, *radius, &color.to_paint());
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
                canvas.draw_image(image, self.position, None);
            }
            WidgetData::TextPane { text_pane } => {


                let jungle_green = Color::parse_hex("#62b4a6");
                let eggplant = Color::parse_hex("#530922");

                canvas.save();
                canvas.clip_rect(self.bounding_rect(), None, None);
                let rrect = RRect::new_rect_xy(self.bounding_rect(), 20.0, 20.0);
                canvas.draw_rrect(rrect, &eggplant.to_paint());
                let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(), 32.0);


                canvas.clip_rect(self.bounding_rect().with_inset((20,20)), None, None);
                let fractional_offset = text_pane.fractional_line_offset();
                canvas.translate((self.position.x + 30.0 - text_pane.offset.x, self.position.y + text_pane.line_height - fractional_offset + 10.0));

                for line in text_pane.visible_lines(self.size.height) {
                    canvas.draw_str(line, Point::new(0.0, 0.0), &font, &jungle_green.to_paint());
                    canvas.translate((0.0, text_pane.line_height));
                }

                canvas.restore();
            }
            WidgetData::Text { text, text_options } => {

                let font = Font::new(Typeface::new(text_options.font_family.clone(), text_options.font_weight.into()).unwrap(), text_options.size);
                let paint = text_options.color.to_paint();
                canvas.draw_str(text, (self.position.x, self.position.y), &font, &paint);
            }
            WidgetData::Process { process } => {
                let file_name = process.file_path.file_name().unwrap().to_str().unwrap();
                let font = Font::new(Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(), 32.0);
                let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
                canvas.draw_str(file_name, Point::new(self.position.x, self.position.y), &font, white);
            }
            WidgetData::HoverFile { path: _ } => {
                let purple = Color::parse_hex("#1c041e");
                canvas.draw_rect(self.bounding_rect(), &purple.to_paint());
            }

        }
    }

    pub fn mouse_over(&self, position: &Position) -> bool {
        let x = position.x;
        let y = position.y;
        let x_min = self.position.x;
        let x_max = self.position.x + self.size.width;
        let y_min = self.position.y;
        let y_max = self.position.y + self.size.height;
        x >= x_min && x <= x_max && y >= y_min && y <= y_max
    }
}


// TODO: I might need tags or things like that
// I need a much richer notion for widgets if I'm going
// to let you select them. Yes I know I'm recreating a browser
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum WidgetSelector {
    This,
    ById(WidgetId),
    ByName(String),
    ByArea{ x: f32, y: f32, width: f32, height: f32 },
}


impl WidgetSelector {
    // TODO: In order for this to work I need to track
    // the originator of the event and pass it here.
    pub fn select(&self, _widgets: &WidgetStore) -> Vec<WidgetId> {
        match self {
            WidgetSelector::This => {
                todo!("Need to track event originator");
            }
            WidgetSelector::ById(id) => {
                vec![*id]
            }
            WidgetSelector::ByName(_) => {
                todo!("By name selector");
            }
            WidgetSelector::ByArea{ .. } => {
                todo!("By area selector");
            }
        }
    }
}
