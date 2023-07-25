use std::{
    cell::RefCell, collections::HashMap, fs::File, io::Read, path::PathBuf, process::ChildStdout,
    str::from_utf8,
};

use nonblock::NonBlockingReader;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use skia_safe::{
    font_style::{Slant, Weight, Width},
    Canvas, Color4f, Data, Font, FontStyle, Image, Paint, Path, Point, RRect, Rect, Typeface,
};

use crate::{
    editor::Value,
    event::Event,
    wasm_messenger::{self, WasmId, WasmMessenger},
};

#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}
impl Position {
    pub fn offset(&self, x: f32, y: f32) -> Position {
        Position {
            x: self.x + x,
            y: self.y + y,
        }
    }
}

impl From<Position> for Point {
    fn from(val: Position) -> Self {
        Point { x: val.x, y: val.y }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
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
    pub scale: f32,
    // Children might make sense
    // pub children: Vec<Widget>,
    pub data: WidgetData,
    #[serde(default)]
    pub ephemeral: bool,
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
        self.widgets.get(id)
    }

    pub fn get_mut(&mut self, id: usize) -> Option<&mut Widget> {
        self.widgets.get_mut(id)
    }

    pub fn new() -> WidgetStore {
        WidgetStore {
            widgets: Vec::new(),
            next_id: 0,
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Widget> {
        self.widgets.iter_mut().filter(|x| !x.data.is_deleted())
    }

    pub fn iter(&self) -> impl Iterator<Item = &Widget> {
        self.widgets.iter().filter(|x| !x.data.is_deleted())
    }

    pub fn clear(&mut self) {
        self.next_id = 0;
        self.widgets.clear();
    }

    pub fn delete_widget(&mut self, widget_id: usize) {
        // TODO: Need a better way rather than tombstones
        self.widgets[widget_id].data = WidgetData::Deleted;
    }

    pub fn get_widget_by_wasm_id(&self, expected_wasm_id: usize) -> Option<usize> {
        self.widgets
            .iter()
            .find(|x| match &x.data {
                WidgetData::Wasm { wasm: _, wasm_id } => *wasm_id as usize == expected_wasm_id,
                _ => false,
            })
            .map(|x| x.id)
    }
}

#[derive(Copy, Clone, Serialize, Deserialize)]
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

    #[allow(unused)]
    pub fn to_sk_color(&self) -> skia_safe::Color {
        self.to_color4f().to_color()
    }

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    #[allow(unused)]
    pub fn from_color4f(color4f: &Color4f) -> Self {
        Color {
            r: color4f.r,
            g: color4f.g,
            b: color4f.b,
            a: color4f.a,
        }
    }

    pub fn parse_hex(hex: &str) -> Color {
        let mut start = 0;
        if hex.starts_with('#') {
            start = 1;
        }

        let r = i64::from_str_radix(&hex[start..start + 2], 16).unwrap() as f32;
        let g = i64::from_str_radix(&hex[start + 2..start + 4], 16).unwrap() as f32;
        let b = i64::from_str_radix(&hex[start + 4..start + 6], 16).unwrap() as f32;
        Color::new(r / 255.0, g / 255.0, b / 255.0, 1.0)
    }
}

// I could go the interface route here.
// I like enums. Will consider it later.
#[derive(Serialize, Deserialize)]
pub enum WidgetData {
    Circle {
        radius: f32,
        color: Color,
    },
    Compound {
        children: Vec<WidgetId>,
    },
    Image {
        data: ImageData,
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
    Wasm {
        wasm: Wasm,
        #[serde(skip)]
        wasm_id: WasmId,
    },
    // I probably don't need this
    // But I think no with my mouse fix
    // I could do something cool with it?
    HoverFile {
        path: String,
    },
    Deleted,
}

impl WidgetData {
    pub fn is_deleted(&self) -> bool {
        match self {
            WidgetData::Deleted => true,
            _ => false,
        }
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
    stdout: Option<NonBlockingReader<ChildStdout>>,
}

impl Process {
    pub fn _new(file_path: PathBuf) -> Process {
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

impl From<FontWeight> for FontStyle {
    fn from(val: FontWeight) -> Self {
        match val {
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

pub fn encode_base64(data: &str) -> String {
    use base64::Engine;

    base64::engine::general_purpose::STANDARD.encode(data)
}

pub fn decode_base64(data: &Vec<u8>) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    use base64::Engine;
    let data = base64::engine::general_purpose::STANDARD.decode(data)?;
    Ok(data)
}

fn serialize_text<S>(x: &Vec<u8>, s: S) -> Result<S::Ok, S::Error>
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
pub struct TextPane {
    #[serde(serialize_with = "serialize_text")]
    #[serde(deserialize_with = "deserialize_text")]
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

    fn visible_lines(&self, height: f32) -> impl std::iter::Iterator<Item = &str> + '_ {
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

    fn fractional_line_offset(&self) -> f32 {
        self.offset.y % self.line_height
    }

    pub fn set_text(&mut self, output: &str) {
        self.contents = output.into();
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
    pub fn _new(path: String) -> Self {
        Self {
            path,
            cache: RefCell::new(None),
        }
    }

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
        let image = Image::from_encoded(Data::new_copy(image_data.as_ref())).unwrap();
        self.cache.replace(Some(image));
    }
}

// I need to also keep track of how to recompile
// things at some point.

#[derive(Serialize, Deserialize)]
pub struct Wasm {
    pub path: String,
    state: Option<String>,
    partial_state: Option<String>,
}

impl Wasm {
    pub fn new(path: String) -> Self {
        Self { path, state: None, partial_state: None }
    }
    
}

impl Widget {
    fn bounding_rect(&self) -> Rect {
        Rect::from_xywh(
            self.position.x,
            self.position.y,
            self.size.width,
            self.size.height,
        )
    }

    pub fn on_click(
        &mut self,
        position: &Position,
        wasm_messenger: &mut WasmMessenger,
    ) -> Vec<Event> {
        let widget_x = position.x - self.position.x;
        let widget_y = position.y - self.position.y;
        let widget_space = Position {
            x: widget_x,
            y: widget_y,
        };
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_click(*wasm_id, &widget_space);
                vec![]
            }
            _ => self.on_click.clone(),
        }
    }

    pub fn draw(
        &mut self,
        canvas: &mut Canvas,
        wasm_messenger: &mut WasmMessenger,
        bounds: Size,
        #[allow(unused)] values: &HashMap<String, Value>,
    ) -> Vec<WidgetId> {
        canvas.save();
        // Have to do this to deal with mut stuff
        if let WidgetData::Wasm { wasm: _, wasm_id } = &mut self.data {
            canvas.save();
            canvas.translate((self.position.x, self.position.y));
            canvas.clip_rect(Rect::from_wh(bounds.width, bounds.height), None, false);
            canvas.scale((self.scale, self.scale));
            wasm_messenger.draw_widget(*wasm_id, canvas, bounds);
            // if let Some(widget_size) = widget_size {
            //     println!("widget size: {:?}", widget_size);
            // };

            // if let Some(size) = wasm_messenger.draw_widget(*wasm_id, canvas, bounds) {
            //     self.size = size;
            // }
            canvas.translate((self.size.width, 0.0));
            // wasm.draw_debug(canvas);
            // if let Some(size) = wasm.draw_debug(canvas) {
            //     self.size.width += size.width;
            //     self.size.height += size.height;
            // }
            canvas.restore();
        }

        match &self.data {
            // TODO: Get rid of compound
            WidgetData::Compound { children } => {
                canvas.restore();
                return children.clone();
            }
            WidgetData::Image { data } => {
                canvas.save();
                canvas.scale((self.scale, self.scale));
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
                canvas.restore();
            }
            WidgetData::TextPane { text_pane } => {
                canvas.save();
                let text_pane = &text_pane;
                let foreground = Color::parse_hex("#dc9941");
                let background = Color::parse_hex("#353f38");

                let paint = background.to_paint();
                canvas.save();
                canvas.translate((self.position.x, self.position.y));
                canvas.clip_rect(Rect::from_wh(bounds.width, bounds.height), None, false);
                canvas.scale((self.scale, self.scale));

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

                for line in text_pane.visible_lines(self.size.height) {
                    canvas.draw_str(line, Point::new(0.0, 0.0), &font, &foreground.to_paint());
                    canvas.translate((0.0, text_pane.line_height));
                }

                canvas.restore();
                canvas.restore();
            }
            WidgetData::Text { text, text_options } => {
                canvas.save();
                canvas.scale((self.scale, self.scale));
                let font = Font::new(
                    Typeface::new(
                        text_options.font_family.clone(),
                        text_options.font_weight.into(),
                    )
                    .unwrap(),
                    text_options.size,
                );
                let paint = text_options.color.to_paint();
                canvas.draw_str(
                    text,
                    (self.position.x, self.position.y + self.size.height),
                    &font,
                    &paint,
                );
                canvas.restore();
            }
            WidgetData::Process { process } => {
                canvas.save();
                canvas.scale((self.scale, self.scale));
                let file_name = process.file_path.file_name().unwrap().to_str().unwrap();
                let font = Font::new(
                    Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(),
                    32.0,
                );
                let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);
                canvas.draw_str(
                    file_name,
                    Point::new(self.position.x, self.position.y),
                    &font,
                    white,
                );
                canvas.restore();
            }
            WidgetData::HoverFile { path: _ } => {
                canvas.save();
                canvas.scale((self.scale, self.scale));
                let purple = Color::parse_hex("#1c041e");
                canvas.draw_rect(self.bounding_rect(), &purple.to_paint());
                canvas.restore();
            }

            _ => {}
        }
        canvas.restore();
        vec![]
    }

    pub fn mouse_over(&self, position: &Position) -> bool {
        let x = position.x;
        let y = position.y;
        let x_min = self.position.x;
        let x_max = self.position.x + self.size.width * self.scale;
        let y_min = self.position.y;
        let y_max = self.position.y + self.size.height * self.scale;
        x >= x_min && x <= x_max && y >= y_min && y <= y_max
    }

    pub fn init(&mut self, wasm_messenger: &mut WasmMessenger) {
        match &mut self.data {
            WidgetData::Wasm { wasm, wasm_id } => {
                let new_wasm_id = wasm_messenger.new_instance(&wasm.path, None);
                *wasm_id = new_wasm_id;
                if let Some(state) = &wasm.state {
                    wasm_messenger.send_set_state(*wasm_id, state);
                }
            }
            _ => {}
        }
    }

    pub fn save(&mut self, wasm_messenger: &mut WasmMessenger) {
        if self.ephemeral {
            return;
        }
        match &mut self.data {
            WidgetData::Wasm { wasm, wasm_id } => match wasm_messenger.save_state(*wasm_id) {
                wasm_messenger::SaveState::Unsaved => {
                    panic!("Wasm instance {} is unsaved", wasm_id)
                }
                wasm_messenger::SaveState::Empty => {
                    wasm.state = None;
                }
                wasm_messenger::SaveState::Saved(state) => {
                    wasm.state = Some(state);
                }
            },
            _ => {}
        }
    }

    pub fn files_to_watch(&self) -> Vec<String> {
        match &self.data {
            WidgetData::Wasm { wasm, .. } => {
                vec![wasm.path.clone()]
            }
            WidgetData::Process { process } => {
                vec![process.file_path.to_str().unwrap().to_string()]
            }
            _ => {
                vec![]
            }
        }
    }

    pub fn send_process_message(
        &self,
        process_id: usize,
        buf: &str,
        wasm_messenger: &mut WasmMessenger,
    ) {
        match &self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_process_message(*wasm_id, process_id, buf);
            }
            _ => {
                panic!("Can't send process message to non-wasm widget");
            }
        }
    }

    pub fn on_size_change(&mut self, width: f32, height: f32, wasm_messenger: &mut WasmMessenger) {
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_size_change(*wasm_id, width, height);
            }
            _ => {}
        }
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
    ByArea {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    },
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
            WidgetSelector::ByArea { .. } => {
                todo!("By area selector");
            }
        }
    }
}
