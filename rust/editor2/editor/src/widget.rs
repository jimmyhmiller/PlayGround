use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
    fs::File,
    io::Read,
    path::PathBuf,
};


use serde::{Deserialize, Serialize};
use skia_safe::{
    font_style::{Slant, Weight, Width},
    Canvas, Data, Font, FontStyle, Image, Point, Typeface,
};

use crate::{
    event::Event,
    wasm_messenger::{self, WasmId, WasmMessenger}, keyboard::KeyboardInput, widget2::{Widget as Widget2, TextPane, WasmWidget}, color::Color,
};



#[derive(Copy, Clone, Serialize, Deserialize, Debug, PartialEq, PartialOrd)]
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

impl From<Size> for Position {
    fn from(val: Size) -> Self {
        Position {
            x: val.width,
            y: val.height,
        }
    }
}

pub type WidgetId = usize;


// use crate::widget2::{widget_serialize, widget_deserialize};
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
    #[serde(skip)]
    // #[serde(serialize_with = "widget_serialize", deserialize_with = "widget_deserialize")]
    pub data2: Box<dyn Widget2>,
    #[serde(default)]
    pub ephemeral: bool,
}

pub struct WidgetStore {
    widgets: Vec<Widget>,
    widget_images: HashMap<WidgetId, Image>,
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
            widget_images: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn draw(
        &mut self,
        canvas: &Canvas,
        wasm_messenger: &mut WasmMessenger,
        dirty_widgets: &HashSet<usize>,
    ) {
        let mut dirty_widgets = dirty_widgets.clone();
        for widget in self.iter() {
            if !self.widget_images.contains_key(&widget.id) {
                dirty_widgets.insert(widget.id);
            }
        }
        let mut images_to_insert = vec![];
        for widget_id in dirty_widgets.iter() {
            if let Some(widget) = self.get_mut(*widget_id) {
                if let Some(mut surface) = canvas.new_surface(&canvas.image_info(), None) {
                    let canvas = surface.canvas();

                    let before_count = canvas.save();

                    // TODO: Still broken because of dirty checking
                    // but we are drawing 

                    // let can_draw = match widget.data {
                    //     WidgetData::Wasm { wasm: _, wasm_id } => {
                    //         widget.data2.has_draw_commands(wasm_id)
                    //     }
                    //     _ => true,
                    // };

                    // if !can_draw {
                    //     continue;
                    // }

                    widget.draw(canvas, wasm_messenger, widget.size);
                    canvas.restore_to_count(before_count);
                    canvas.restore();

                    let image = surface.image_snapshot();
                    images_to_insert.push((widget.id, image));
                }
            } else {
                println!("Widget not found for id: {}", widget_id);
            }
        }

        for (id, image) in images_to_insert {
            self.widget_images.insert(id, image);
        }

        for (_, image) in self.widget_images.iter() {
            canvas.draw_image(image, (0.0, 0.0), None);
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
        self.widget_images.remove(&widget_id);
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

// I could go the interface route here.
// I like enums. Will consider it later.
#[derive(Serialize, Deserialize)]
pub enum WidgetData {
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
    Wasm {
        wasm: Wasm,
        #[serde(skip)]
        wasm_id: WasmId,
    },
    Deleted,
}

impl WidgetData {
    pub fn is_deleted(&self) -> bool {
        matches!(self, WidgetData::Deleted)
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


#[derive(Serialize, Deserialize)]
pub struct Wasm {
    pub path: String,
    state: Option<String>,
    partial_state: Option<String>,
}

impl Wasm {
    pub fn new(path: String) -> Self {
        Self {
            path,
            state: None,
            partial_state: None,
        }
    }
}

impl Widget {

    pub fn on_click(
        &mut self,
        position: &Position,
        wasm_messenger: &mut WasmMessenger,
    ) -> Vec<Event> {
        let widget_space = self.widget_space(position);
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_click(*wasm_id, &widget_space);
                vec![]
            }
            _ => self.on_click.clone(),
        }
    }

    fn widget_space(&mut self, position: &Position) -> Position {
        let widget_x = position.x - self.position.x;
        let widget_y = position.y - self.position.y;

        Position {
            x: widget_x,
            y: widget_y,
        }
    }
    

    pub fn on_mouse_down(&mut self, position: &Position, wasm_messenger: &mut WasmMessenger) {
        let widget_space = self.widget_space(position);
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_mouse_down(*wasm_id, &widget_space);
            }
            _ => {}
        }
    }

    pub fn on_mouse_up(&mut self, position: &Position, wasm_messenger: &mut WasmMessenger) {
        let widget_space = self.widget_space(position);
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_mouse_up(*wasm_id, &widget_space);
            }
            _ => {}
        }
    }

    pub fn draw(
        &mut self,
        canvas: &Canvas,
        wasm_messenger: &mut WasmMessenger,
        bounds: Size,
    ) -> Vec<WidgetId> {
        canvas.save();
        // Have to do this to deal with mut stuff
        if let WidgetData::Wasm { wasm: _, wasm_id } = &mut self.data {
            canvas.save();
            canvas.translate((self.position.x, self.position.y));
            canvas.scale((self.scale, self.scale));
            // wasm_messenger.draw_widget(*wasm_id, canvas, bounds);
            self.data2.draw(canvas, bounds).unwrap();
            canvas.translate((self.size.width, 0.0));
            canvas.restore();
        }

        match &self.data {
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
            WidgetData::TextPane { text_pane: _ } => {
                self.data2.draw(canvas, bounds).unwrap();
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
                let paint = text_options.color.as_paint();
                canvas.draw_str(
                    text,
                    (self.position.x, self.position.y + self.size.height),
                    &font,
                    &paint,
                );
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
                let (new_wasm_id, receiver) = wasm_messenger.new_instance(&wasm.path, None);

                self.data2 = Box::new(WasmWidget {
                    draw_commands: vec![],
                    sender: wasm_messenger.get_sender(new_wasm_id),
                    receiver,
                });
                *wasm_id = new_wasm_id;
                if let Some(state) = &wasm.state {
                    self.data2.set_state(state.clone()).unwrap();
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
            WidgetData::Deleted => {}
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

    pub fn on_move(&mut self, x: f32, y: f32, wasm_messenger: &mut WasmMessenger) {
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_move(*wasm_id, x, y);
            }
            WidgetData::TextPane { .. } => {
                self.data2.on_move(x, y).unwrap();
            }
            _ => {}
        }
    }

    pub fn on_scroll(&mut self, x: f64, y: f64, wasm_messenger: &mut WasmMessenger) -> bool {
        match &mut self.data {
            WidgetData::TextPane { .. } => {
                self.data2.on_scroll(x, y).unwrap();
                true
            }
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_scroll(*wasm_id, x, y);
                true
            }
            _ => {
                false
            }
        }
    }

    pub fn on_event(&mut self, kind: &str, event: &str, wasm_messenger: &mut WasmMessenger) -> bool {
        match &mut self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_event(
                    *wasm_id,
                    kind.to_string(),
                    event.to_string(),
                );
                true
            }
            _ => { 
                false
            }
        }
    }

    pub fn on_key(&mut self, input: KeyboardInput, wasm_messenger: &mut WasmMessenger) -> bool {
        match self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_key(
                    wasm_id,
                    input,
                );
                true
            }
            _ => {
                false
            }
        }
    }

    pub fn on_mouse_move(&self, widget_space: &Position, x_diff: f32, y_diff: f32, wasm_messenger: &mut WasmMessenger) -> bool {
        match &self.data {
            WidgetData::Wasm { wasm: _, wasm_id } => {
                wasm_messenger.send_on_mouse_move(
                    *wasm_id,
                    widget_space,
                    x_diff,
                    y_diff,
                );
                true
            }
            _ => {
                false
            }
        }
    }
}
