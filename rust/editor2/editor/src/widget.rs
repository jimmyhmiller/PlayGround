use std::{
    cell::RefCell,
    collections::{HashMap, HashSet},
};


use serde::{Deserialize, Serialize};
use skia_safe::{
    font_style::{Slant, Weight, Width},
    Canvas, FontStyle, Image, Point,
};

use crate::{
    wasm_messenger::{self, WasmId, WasmMessenger, SaveState}, keyboard::KeyboardInput, widget2::{Widget as Widget2, TextPane, WasmWidget, WidgetMeta, Text, self}, color::Color,
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
    pub scale: f32,
    // Children might make sense
    // pub children: Vec<Widget>,
    pub data: WidgetData,
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

                    widget.draw(canvas, widget.size);
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

#[derive(Clone, Serialize, Deserialize)]
pub struct TextOptions {
    pub font_family: String,
    pub font_weight: FontWeight,
    pub size: f32,
    pub color: Color,
}

#[derive(Serialize, Deserialize)]
pub struct ImageData {
    path: String,
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

    pub fn on_click(&mut self, position: &Position) {
        let widget_space = self.widget_space(position);
        self.data2.on_click(widget_space.x, widget_space.y).unwrap();
    }

    fn widget_space(&mut self, position: &Position) -> Position {
        let widget_x = position.x - self.position.x;
        let widget_y = position.y - self.position.y;

        Position {
            x: widget_x,
            y: widget_y,
        }
    }
    

    pub fn on_mouse_down(&mut self, position: &Position) {
        let widget_space = self.widget_space(position);
        match &mut self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_mouse_down(widget_space.x, widget_space.y).unwrap();
            }
            _ => {}
        }
    }

    pub fn on_mouse_up(&mut self, position: &Position) {
        let widget_space = self.widget_space(position);
        match &mut self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_mouse_up(widget_space.x, widget_space.y).unwrap();
            }
            _ => {}
        }
    }

    pub fn draw(
        &mut self,
        canvas: &Canvas,
        bounds: Size,
    ) -> Vec<WidgetId> {
        canvas.save();
        self.data2.draw(canvas, bounds).unwrap();
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
                    sender: Some(wasm_messenger.get_sender(new_wasm_id)),
                    receiver: Some(receiver),
                    meta: WidgetMeta::new(self.position, self.size, self.scale),
                    save_state: SaveState::Unsaved,
                });
                *wasm_id = new_wasm_id;
                if let Some(state) = &wasm.state {
                    self.data2.set_state(state.clone()).unwrap();
                }
            }
            WidgetData::Text { text, text_options } => {
                self.data2 = Box::new(Text {
                    text: text.clone(),
                    text_options: text_options.clone(),
                    meta: WidgetMeta::new(self.position, self.size, self.scale),
                });
            }
            WidgetData::TextPane { text_pane } => {
                self.data2 = Box::new(text_pane.clone());
            }
            WidgetData::Image { data } => {
                self.data2 = Box::new(widget2::Image {
                    path: data.path.clone(),
                    cache: RefCell::new(None),
                    meta: WidgetMeta::new(self.position, self.size, self.scale),
                });
            }
            _ => {}
        }
    }

    pub fn save(&mut self, wasm_messenger: &mut WasmMessenger) {
        if self.ephemeral {
            return;
        }
        // TODO: Clean up this mess
        loop {
            match &mut self.data {
                WidgetData::Wasm { wasm, .. } => {

                    self.data2.save().unwrap();
                    wasm_messenger.tick(&mut HashMap::new());
                    self.data2.update().unwrap();
                    let wasm_widget : &WasmWidget = self.data2.as_any().downcast_ref().unwrap();
                    match &wasm_widget.save_state {
                        wasm_messenger::SaveState::Unsaved => {
                            continue;
                        }
                        wasm_messenger::SaveState::Empty => {
                            wasm.state = None;
                            break;
                        }
                        wasm_messenger::SaveState::Saved(state) => {
                            wasm.state = Some(state.clone());
                            break;
                        }
                    }
                }
                _ => {
                    break;
                }
            }
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
        &mut self,
        process_id: usize,
        buf: &str,
    ) {
        match &self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_process_message(process_id as i32, buf.to_string()).unwrap();
            }
            WidgetData::Deleted => {}
            _ => {
                panic!("Can't send process message to non-wasm widget");
            }
        }
    }

    pub fn on_size_change(&mut self, width: f32, height: f32) {
        match &mut self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_size_change(width, height).unwrap();
            }
            _ => {}
        }
    }

    pub fn on_move(&mut self, x: f32, y: f32) {
        match &mut self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_move(x, y).unwrap();
            }
            WidgetData::TextPane { .. } => {
                self.data2.on_move(x, y).unwrap();
            }
            _ => {}
        }
    }

    pub fn on_scroll(&mut self, x: f64, y: f64) -> bool {
        match &mut self.data {
            WidgetData::TextPane { .. } => {
                self.data2.on_scroll(x, y).unwrap();
                true
            }
            WidgetData::Wasm { .. } => {
                self.data2.on_scroll(x, y).unwrap();
                true
            }
            _ => {
                false
            }
        }
    }

    pub fn on_event(&mut self, kind: &str, event: &str) -> bool {
        match &mut self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_event(kind.to_string(), event.to_string()).unwrap();
                true
            }
            _ => { 
                false
            }
        }
    }

    pub fn on_key(&mut self, input: KeyboardInput) -> bool {
        match self.data {
            WidgetData::Wasm { .. } => {
               self.data2.on_key(input.to_framework()).unwrap();
                true
            }
            _ => {
                false
            }
        }
    }

    pub fn on_mouse_move(&mut self, widget_space: &Position, x_diff: f32, y_diff: f32) -> bool {
        match &self.data {
            WidgetData::Wasm { .. } => {
                self.data2.on_mouse_move(widget_space.x, widget_space.y, x_diff, y_diff).unwrap();
                true
            }
            _ => {
                false
            }
        }
    }
}
