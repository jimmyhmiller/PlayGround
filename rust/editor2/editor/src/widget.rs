use std::{
    collections::{HashMap, HashSet},
    ops::{Deref, DerefMut},
};

use framework::{Position, Value, WidgetMeta};
use itertools::Itertools;
use serde::{Deserialize, Serialize};
use skia_safe::{Canvas, Image};

use crate::{
    event::Event,
    keyboard::KeyboardInput,
    wasm_messenger::{self, SaveState, WasmMessenger},
    widget2::{self, Widget as Widget2},
};

pub type WidgetId = usize;

#[derive(Serialize, Deserialize)]
pub struct Widget {
    pub data: Box<dyn Widget2>,
}
impl Deref for Widget {
    type Target = dyn Widget2;

    fn deref(&self) -> &Self::Target {
        self.data.deref()
    }
}

impl DerefMut for Widget {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data.deref_mut()
    }
}

impl Widget {
    pub fn on_click(&mut self, position: &Position) {
        let widget_space = self.widget_space(position);
        self.data.on_click(widget_space.x, widget_space.y).unwrap();
    }

    pub fn on_delete(&mut self) {
        self.data.on_delete().unwrap();
    }

    fn widget_space(&mut self, position: &Position) -> Position {
        let widget_x = position.x - self.position().x;
        let widget_y = position.y - self.position().y;

        Position {
            x: widget_x,
            y: widget_y,
        }
    }

    pub fn on_mouse_down(&mut self, position: &Position) {
        let widget_space = self.widget_space(position);
        self.data
            .on_mouse_down(widget_space.x, widget_space.y)
            .unwrap();
    }

    pub fn on_mouse_up(&mut self, position: &Position) {
        let widget_space = self.widget_space(position);
        self.data
            .on_mouse_up(widget_space.x, widget_space.y)
            .unwrap();
    }

    pub fn draw(&mut self, canvas: &Canvas) -> Vec<WidgetId> {
        canvas.save();
        self.data.draw(canvas).unwrap();
        canvas.restore();
        vec![]
    }

    pub fn mouse_over(&self, position: &Position) -> bool {
        let x = position.x;
        let y = position.y;
        let x_min = self.position().x;
        let x_max = x_min + (self.size().width * self.scale());
        let y_min = self.position().y;
        let y_max = y_min + (self.size().height * self.scale());
        x >= x_min && x <= x_max && y >= y_min && y <= y_max
    }

    pub fn init(
        &mut self,
        wasm_messenger: &mut WasmMessenger,
        values: HashMap<String, Value>,
        external_sender: std::sync::mpsc::Sender<Event>,
    ) {
        if let Some(widget) = self.as_wasm_widget_mut() {
            let (new_wasm_id, receiver) = wasm_messenger.new_instance(
                &widget.path,
                None,
                values,
                external_sender,
                widget.id(),
            );
            widget.sender = Some(wasm_messenger.get_sender(new_wasm_id));
            widget.receiver = Some(receiver);
            match &widget.save_state {
                SaveState::Unsaved => println!("Unsaved!!!"),
                SaveState::Empty => println!("Empty"),
                SaveState::Saved(state) => {
                    widget
                        .set_state(serde_json::to_string(state).unwrap())
                        .unwrap();
                }
            }
        }
        self.start().unwrap();
    }

    pub fn save(&mut self, wasm_messenger: &mut WasmMessenger)  {
        // TODO: Change this
        if self.data.typetag_name() == "Ephemeral" {
            return;
        }
        // TODO: Clean up this mess
        while let Some(widget) = self.as_wasm_widget_mut() {
            widget.save().unwrap();
            wasm_messenger.tick();
            widget.update().unwrap();
            match &widget.save_state {
                wasm_messenger::SaveState::Unsaved => {
                    continue;
                }
                wasm_messenger::SaveState::Empty => {
                    break;
                }
                wasm_messenger::SaveState::Saved(value) => {
                    break;
                }
            }
        }
    }

    pub fn files_to_watch(&self) -> Vec<String> {
        if let Some(widget) = self.as_wasm_widget() {
            vec![widget.path.clone()]
        } else {
            vec![]
        }
    }

    pub fn send_process_message(&mut self, process_id: usize, buf: &str) {
        self.data
            .on_process_message(process_id as i32, buf.to_string())
            .unwrap();
    }

    pub fn on_size_change(&mut self, width: f32, height: f32) {
        self.data.on_size_change(width, height).unwrap();
    }

    pub fn on_move(&mut self, x: f32, y: f32) {
        self.data.on_move(x, y).unwrap();
    }

    pub fn on_scroll(&mut self, x: f64, y: f64) -> bool {
        self.data.on_scroll(x, y).unwrap();
        true
    }

    pub fn on_event(&mut self, kind: &str, event: &str) -> bool {
        self.data
            .on_event(kind.to_string(), event.to_string())
            .unwrap();
        true
    }

    pub fn on_key(&mut self, input: KeyboardInput) -> bool {
        self.data.on_key(input.as_framework()).unwrap();
        true
    }

    pub fn on_mouse_move(&mut self, widget_space: &Position, x_diff: f32, y_diff: f32) -> bool {
        self.data
            .on_mouse_move(widget_space.x, widget_space.y, x_diff, y_diff)
            .unwrap();
        true
    }

    pub fn meta(&self) -> WidgetMeta {
        WidgetMeta::new(
            self.position(),
            self.size(),
            self.scale(),
            self.id(),
            self.typetag_name().to_string(),
        )
    }
}

pub struct WidgetStore {
    widgets: Vec<Widget>,
    widget_images: HashMap<WidgetId, Image>,
    next_id: WidgetId,
    z_indexes: Vec<usize>,
    should_cache_draw: bool,
}

impl WidgetStore {
    pub fn next_id(&mut self) -> WidgetId {
        let current = self.next_id;
        self.next_id += 1;
        current
    }
    pub fn add_widget(&mut self, widget: Widget) -> WidgetId {
        let id = widget.id();
        self.widgets.push(widget);
        if id + 1 != self.widgets.len() {
            println!("not equal {} {}", id, self.widgets.len());
        }
        self.z_indexes.push(id);
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
            z_indexes: Vec::new(),
            should_cache_draw: false,
        }
    }

    pub fn draw(&mut self, canvas: &Canvas, dirty_widgets: &HashSet<usize>) {
        // TODO: Created Widgets don't draw right away

        // TODO: Deleted widgets can end up here

        // let dirty_widgets: HashSet<usize> = self.widgets.iter().map(|x| x.id()).collect();
        let mut dirty_widgets = dirty_widgets.clone();
        for widget in self.iter() {
            if !self.widget_images.contains_key(&widget.id()) {
                dirty_widgets.insert(widget.id());
            }
        }
        let mut images_to_insert: Vec<(usize, Image)> = vec![];

        if !self.should_cache_draw {
            for id in self.z_indexes.clone().iter() {
                if let Some(widget) = self.widgets.get_mut(*id) {
                    let before_count = canvas.save();
                    canvas.translate((widget.position().x, widget.position().y));
                    widget.draw(canvas);
                    canvas.restore_to_count(before_count);
                    canvas.restore();
                }
            }
            return;
        }
        for widget in self.widgets.iter_mut() {
            if !dirty_widgets.contains(&widget.id()) {
                continue;
            }

            let image_info = canvas.image_info();
            let image_info = image_info
                .with_dimensions((widget.size().width as i32, widget.size().height as i32));

            if let Some(mut surface) = canvas.new_surface(&image_info, None) {
                let canvas = surface.canvas();

                let before_count = canvas.save();
                canvas.translate((0, 0));

                // TODO: Still broken because of dirty checking
                // but we are drawing

                let can_draw = if let Some(widget) = widget.as_wasm_widget() {
                    !widget.draw_commands.is_empty()
                } else {
                    true
                };

                if !can_draw {
                    continue;
                }
                widget.draw(canvas);
                canvas.restore_to_count(before_count);
                canvas.restore();

                let image = surface.image_snapshot();
                images_to_insert.push((widget.id(), image));
            } else {
                // TODO: This is wrong for z-indexing
                // I could chunk up this canvas and draw each chunk
                canvas.save();
                canvas.translate((widget.position().x, widget.position().y));
                widget.draw(canvas);
                canvas.restore();
                self.widget_images.remove(&widget.id());
            }
        }

        for (id, image) in images_to_insert {
            self.widget_images.insert(id, image);
        }

        for id in self.z_indexes.iter() {
            if let Some(image) = self.widget_images.get(id) {
                if let Some(widget) = self.get(*id) {
                    canvas.draw_image(image, (widget.position().x, widget.position().y), None);
                }
            }
        }
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut Widget> {
        self.widgets.iter_mut()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Widget> {
        self.widgets.iter()
    }

    #[allow(unused)]
    pub fn iter_by_z_index(&self) -> impl Iterator<Item = &Widget> {
        self.z_indexes
            .iter()
            .rev()
            .map(move |x| self.widgets.get(*x).unwrap())
    }

    pub fn iter_by_z_index_mut(&mut self) -> impl Iterator<Item = &mut Widget> {
        pub struct WidgetIterMut<'a> {
            widgets: *mut [Widget],
            indexes: std::vec::IntoIter<usize>,
            _marker: std::marker::PhantomData<&'a mut [Widget]>,
        }

        impl<'a> WidgetIterMut<'a> {
            fn new(widgets: &'a mut [Widget], indexes: Vec<usize>) -> Self {
                Self {
                    widgets: widgets as *mut _,
                    indexes: indexes.into_iter(),
                    _marker: std::marker::PhantomData,
                }
            }
        }

        impl<'a> Iterator for WidgetIterMut<'a> {
            type Item = &'a mut Widget;

            fn next(&mut self) -> Option<Self::Item> {
                let index = self.indexes.next()?;
                unsafe { Some(&mut (*self.widgets)[index]) }
            }
        }
        let mut reverse_indexes = self.z_indexes.clone();
        reverse_indexes.reverse();
        WidgetIterMut::new(&mut self.widgets, reverse_indexes)
    }

    pub fn clear(&mut self) {
        self.next_id = 0;
        self.widgets.clear();
    }

    pub fn delete_widget(&mut self, widget_id: usize) {
        // TODO: Need a better way rather than tombstones
        self.widgets[widget_id].data = Box::new(widget2::Deleted {});
        self.widget_images.remove(&widget_id);
    }

    pub fn on_move(&mut self, widget_id: usize) {
        if let Some((position, _)) = self.z_indexes.iter().find_position(|x| **x == widget_id) {
            // TODO: Swapping is wrong. It moves things in a way that
            // is wrong. We probably need to delete and insert
            self.z_indexes.remove(position);
            self.z_indexes.push(widget_id);
        }
    }

    pub fn fix_zindexes(&mut self) {
        // A widget should always have a greater z-index than its parent

        for widget in self.widgets.iter() {
            if let Some(parent_id) = widget.parent_id() {
                if let Some(z_index_parent) = self.z_indexes.iter().position(|x| *x == parent_id) {
                    if let Some(z_index_child) =
                        self.z_indexes.iter().position(|x| *x == widget.id())
                    {
                        if z_index_parent > z_index_child {
                            self.z_indexes.swap(z_index_parent, z_index_child);
                        }
                    }
                }
            }
        }
    }
}
