#![allow(unused)]
use std::{
    any::Any,
    collections::HashMap,
    ffi::CString,
    fmt::Debug,
    ops::{Deref, DerefMut},
};

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

#[link(wasm_import_module = "host")]
extern "C" {
    #[link_name = "draw_str"]
    fn draw_str_low_level(ptr: i32, len: i32, x: f32, y: f32);
    fn draw_rect(x: f32, y: f32, width: f32, height: f32);
    fn set_color(r: f32, g: f32, b: f32, a: f32);
    fn save();
    fn set_get_state(ptr: u32, len: u32);
    fn clip_rect(x: f32, y: f32, width: f32, height: f32);
    fn draw_rrect(x: f32, y: f32, width: f32, height: f32, radius: f32);
    fn translate(x: f32, y: f32);
    fn restore();
    #[link_name = "set_cursor_icon"]
    fn set_cursor_icon_low_level(cursor_icon: u32);
    fn start_process_low_level(ptr: i32, len: i32) -> u32;
    fn mark_dirty(external_id: u32);
    fn save_file_low_level(path_ptr: i32, path_length: i32, text_ptr: i32, text_length: i32);
    fn send_message_low_level(process_id: u32, ptr: i32, len: i32);
    #[allow(improper_ctypes)]
    fn receive_last_message_low_level(process_id: i32) -> (i32, i32);
    #[link_name = "provide_f32"]
    fn provide_f32_low_level(ptr: i32, len: i32, val: f32);
    #[link_name = "provide_value"]
    fn provide_value_low_level(name_ptr: i32, name_len: i32, ptr: i32, len: i32);
    fn get_value(ptr: i32, len: i32) -> u32;
    fn try_get_value(ptr: i32, len: i32) -> u32;
    #[link_name = "send_event"]
    fn send_event_low_level(kind_ptr: i32, kind_len: i32, ptr: i32, len: i32);
    #[link_name = "subscribe"]
    fn subscribe_low_level(ptr: i32, len: i32);
    #[link_name = "unsubscribe"]
    fn unsubscribe_low_level(ptr: i32, len: i32);
    fn create_widget(x: f32, y: f32, width: f32, height: f32, identifer: u32);
}

#[derive(Clone, Deserialize, Serialize)]
pub struct EventWrapper {
    pub kind: String,
    pub data: String,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PointerLengthString {
    pub ptr: u32,
    pub len: u32,
}

pub type Value = Vec<u8>;

impl From<&String> for PointerLengthString {
    fn from(s: &String) -> Self {
        Self {
            ptr: s.as_ptr() as u32,
            len: s.len() as u32,
        }
    }
}

impl From<PointerLengthString> for String {
    fn from(s: PointerLengthString) -> Self {
        unsafe { String::from_raw_parts(s.ptr as *mut u8, s.len as usize, s.len as usize) }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl Rect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn with_inset(&self, (x, y): (f32, f32)) -> Self {
        // make a rectangle with some margins
        Self {
            x: self.x + x,
            y: self.y + y,
            width: self.width - x * 2.0,
            height: self.height - y * 2.0,
        }
    }
}

pub struct Ui {}

impl Default for Ui {
    fn default() -> Self {
        Self::new()
    }
}
pub enum Component {
    Pane(Size, (f32, f32), Box<Component>),
    List(Vec<Component>, (f32, f32)),
    Container(Box<Component>),
    Text(String),
}

impl Component {
    pub fn draw(&self, canvas: &mut Canvas) {
        match self {
            Component::Pane(size, scroll_offset, child) => {
                let background = Color::parse_hex("#353f38");

                let bounding_rect = Rect::new(0.0, 0.0, size.width, size.height);

                canvas.save();
                canvas.set_color(&background);
                canvas.clip_rect(bounding_rect);
                canvas.draw_rrect(bounding_rect, 20.0);

                canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));
                canvas.translate(scroll_offset.0, scroll_offset.1);
                canvas.translate(0.0, 50.0);
                child.draw(canvas);
            }
            Component::List(children, _scroll_offset) => {
                for child in children.iter() {
                    child.draw(canvas);
                    // TODO: need to translate based on height of component
                    canvas.translate(0.0, 30.0);
                }
            }
            Component::Container(child) => {
                child.draw(canvas);
            }
            Component::Text(text) => {
                canvas.save();
                canvas.set_color(&Color::parse_hex("#ffffff"));
                for line in text.lines() {
                    canvas.draw_str(line, 40.0, 0.0);
                    canvas.translate(0.0, 30.0);
                }
                canvas.restore();
            }
        }
    }
}

impl Ui {
    pub fn new() -> Self {
        Self {}
    }

    pub fn pane(&self, size: Size, scroll_offset: (f32, f32), child: Component) -> Component {
        Component::Pane(size, scroll_offset, Box::new(child))
    }

    pub fn list<Item, I>(
        &self,
        scroll_offset: (f32, f32),
        items: I,
        f: impl Fn(&Self, Item) -> Component,
    ) -> Component
    where
        I: Iterator<Item = Item>,
    {
        // TODO: Make skip and take not be ad-hoc
        let to_skip = (-scroll_offset.1 / 30.0).floor() as usize;
        Component::List(
            items
                .skip(to_skip)
                .take(50)
                .map(|item| f(self, item))
                .collect(),
            scroll_offset,
        )
    }

    pub fn container(&self, child: Component) -> Component {
        Component::Container(Box::new(child))
    }

    pub fn text(&self, text: &str) -> Component {
        Component::Text(text.to_string())
    }
}

pub struct Canvas {
    tranlation: (f32, f32),
    translation_stack: Vec<(f32, f32)>,
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

impl Canvas {
    pub fn new() -> Self {
        Self {
            tranlation: (0.0, 0.0),
            translation_stack: Vec::new(),
        }
    }

    pub fn draw_str(&self, s: &str, x: f32, y: f32) {
        unsafe {
            draw_str_low_level(s.as_ptr() as i32, s.len() as i32, x, y);
        }
    }

    pub fn draw_rect(&self, x: f32, y: f32, width: f32, height: f32) {
        unsafe {
            draw_rect(x, y, width, height);
        }
    }

    pub fn save(&mut self) {
        self.translation_stack.push(self.tranlation);
        unsafe {
            save();
        }
    }

    pub fn clip_rect(&self, rect: Rect) {
        unsafe {
            clip_rect(rect.x, rect.y, rect.width, rect.height);
        }
    }

    pub fn draw_rrect(&self, rect: Rect, radius: f32) {
        unsafe {
            draw_rrect(rect.x, rect.y, rect.width, rect.height, radius);
        }
    }

    pub fn get_current_position(&self) -> (f32, f32) {
        self.tranlation
    }

    pub fn translate(&mut self, x: f32, y: f32) {
        self.tranlation.0 += x;
        self.tranlation.1 += y;
        unsafe {
            translate(x, y);
        }
    }

    pub fn restore(&mut self) {
        if let Some(popped) = self.translation_stack.pop() {
            self.tranlation = popped;
        }

        unsafe {
            restore();
        }
    }

    pub fn set_color(&self, color: &Color) {
        unsafe {
            set_color(color.r, color.g, color.b, color.a);
        }
    }
}

pub static mut DEBUG: Vec<String> = Vec::new();
pub static mut STRING_PTR_TO_LEN: Lazy<HashMap<u32, u32>> = Lazy::new(HashMap::new);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

impl Default for Size {
    fn default() -> Self {
        // We don't want things to be sized to zero by default
        Self {
            width: 100.0,
            height: 100.0,
        }
    }
}

#[derive(Copy, Clone, Serialize, Deserialize, Debug, Default, PartialEq)]
pub struct Position {
    pub x: f32,
    pub y: f32,
}

impl Position {
    pub fn offset(&self, x: f32, y: f32) -> Self {
        Self {
            x: self.x + x,
            y: self.y + y,
        }
    }
    pub fn as_tuple(&self) -> (f32, f32) {
        (self.x, self.y)
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WidgetMeta {
    pub position: Position,
    pub scale: f32,
    pub size: Size,
    #[serde(default)]
    pub id: usize,
    pub parent_id: Option<usize>,
    pub kind: String,
}

impl WidgetMeta {
    pub fn new(position: Position, size: Size, scale: f32, id: usize, kind: String) -> Self {
        Self {
            position,
            scale,
            size,
            id,
            kind,
            parent_id: None,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct WidgetData {
    pub position: Position,
    pub size: Size,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CursorIcon {
    Default = 0,
    Text = 1,
}

impl From<u32> for CursorIcon {
    fn from(i: u32) -> Self {
        match i {
            0 => CursorIcon::Default,
            1 => CursorIcon::Text,
            _ => panic!("Unknown cursor icon"),
        }
    }
}

#[no_mangle]
pub fn alloc_string(len: usize) -> *mut u8 {
    // create a new mutable buffer with capacity `len`
    let mut buf = Vec::with_capacity(len);
    // take a mutable pointer to the buffer
    let ptr = buf.as_mut_ptr();
    // take ownership of the memory block and
    // ensure that its destructor is not
    // called when the object goes out of scope
    // at the end of the function
    std::mem::forget(buf);
    // return the pointer so the runtime
    // can write data at this offset
    unsafe { STRING_PTR_TO_LEN.insert(ptr as u32, len as u32) };
    ptr
}

pub fn fetch_string(str_ptr: u32) -> String {
    let buffer;
    unsafe {
        let len = STRING_PTR_TO_LEN.get(&str_ptr).unwrap();
        buffer = String::from_raw_parts(str_ptr as *mut u8, *len as usize, *len as usize);
        STRING_PTR_TO_LEN.remove(&str_ptr);
    }
    buffer
}

pub fn merge_json(partial_state: Option<String>, state: String) -> String {
    if let Some(partial_state) = partial_state {
        let mut partial_state: serde_json::Value = serde_json::from_str(&partial_state).unwrap();
        let mut state: serde_json::Value = serde_json::from_str(&state).unwrap();

        assert!(partial_state.is_object(), "{:?}", partial_state);
        assert!(state.is_object());

        merge(&mut state, &mut partial_state);
        serde_json::to_string(&state).unwrap()
    } else {
        state
    }
}

pub fn merge(state: &mut serde_json::Value, partial_state: &mut serde_json::Value) {
    match (state, partial_state) {
        (serde_json::Value::Object(state), serde_json::Value::Object(partial_state)) => {
            for (key, value) in partial_state.iter_mut() {
                if let Some(entry) = state.get_mut(key) {
                    merge(entry, value);
                } else {
                    state.insert(key.clone(), value.clone());
                }
            }
        }
        (state, partial_state) => {
            *state = partial_state.clone();
        }
    }
}


pub trait App {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn start(&mut self) {}
    fn draw(&mut self);
    fn on_click(&mut self, x: f32, y: f32);
    fn on_mouse_up(&mut self, x: f32, y: f32) {}
    fn on_mouse_down(&mut self, x: f32, y: f32) {}
    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) {}
    fn on_key(&mut self, input: KeyboardInput);
    fn on_scroll(&mut self, x: f64, y: f64);
    fn on_event(&mut self, kind: String, event: String) {}
    fn on_size_change(&mut self, width: f32, height: f32);
    fn on_move(&mut self, x: f32, y: f32);
    fn on_delete(&mut self) {}
    fn get_position(&self) -> Position;
    fn get_size(&self) -> Size;
    fn get_initial_state(&self) -> String;
    fn get_state(&self) -> String;
    fn set_state(&mut self, state: String);
    fn on_process_message(&mut self, process_id: u32, _message: String) {}
    fn mark_dirty(&self, id: u32) {
        unsafe {
            mark_dirty(id);
        }
    }
}

impl<T: App + ?Sized> AppExtensions for T {}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Widget {
    #[serde(skip)]
    pub app_index: usize,
}

impl Deref for Widget {
    type Target = Box<dyn App>;

    fn deref(&self) -> &Self::Target {
        unsafe { APPS.get(self.app_index).unwrap() }
    }
}

impl DerefMut for Widget {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { APPS.get_mut(self.app_index).unwrap() }
    }
}

// I didn't really need to do this. But it is kind of interesting
pub trait AppExtensions {
    fn start_process(&mut self, process: String) -> u32 {
        unsafe { start_process_low_level(process.as_ptr() as i32, process.len() as i32) }
    }
    
    fn get_external_id(&self) -> Option<usize> {
        unsafe {
            CURRENT_EXTERNAL_ID
        }
    }

    fn get_position2(&self) -> Position {
        unsafe {
            CONTEXT.get(&CURRENT_EXTERNAL_ID).unwrap().meta.position
        }
    }

    fn get_size2(&self) -> Size {
        unsafe {
            CONTEXT.get(&CURRENT_EXTERNAL_ID).unwrap().meta.size
        }
    }

    fn get_scale(&self) -> f32 {
        unsafe {
            CONTEXT.get(&CURRENT_EXTERNAL_ID).unwrap().meta.scale
        }
    }

    fn get_id(&self) -> usize {
        unsafe {
            CONTEXT.get(&CURRENT_EXTERNAL_ID).unwrap().meta.id
        }
    }

    fn create_widget_ref(&mut self, external_id: u32, data: WidgetData) -> Widget {
        unsafe {
            CONTEXT.insert(Some(external_id as usize), Context {
                external_id: Some(external_id as usize),
                meta: WidgetMeta::new(
                    data.position,
                    data.size,
                    1.0,
                    external_id as usize,
                    "widget".to_string(),
                ),
                app_index: self.get_id(),
            });
            
            create_widget(
                data.position.x,
                data.position.y,
                data.size.width,
                data.size.height,
                external_id,
            );
            Widget {
                app_index: external_id as usize,
            }
        }
    }

    fn create_widget(&mut self, app: Box<dyn App>, data: WidgetData) -> Widget {
        unsafe {
            APPS.push(app);
            let identifer = APPS.len() - 1;
            CONTEXT.insert(Some(identifer as usize), Context {
                external_id: Some(identifer as usize),
                meta: WidgetMeta::new(
                    data.position,
                    data.size,
                    1.0,
                    identifer as usize,
                    "widget".to_string(),
                ),
                app_index: identifer,
            });
            
            create_widget(
                data.position.x,
                data.position.y,
                data.size.width,
                data.size.height,
                identifer as u32,
            );
            Widget {
                app_index: identifer,
            }
        }
    }

    // TODO: I need a standard way to send strings to host
    // I have a forget in send_message because I'm pretty sure I needed it
    // but I don't have it elsewhere. I should also probably look into WIT at some point
    fn save_file(&self, path: String, text: String) {
        unsafe {
            save_file_low_level(
                path.as_ptr() as i32,
                path.len() as i32,
                text.as_ptr() as i32,
                text.len() as i32,
            )
        }
        self.send_event("lith/save_file", path);
    }
    fn send_message(&mut self, process_id: u32, message: String) {
        let c_message = CString::new(message).unwrap();
        let ptr = c_message.as_ptr();
        let len = c_message.to_bytes().len();

        unsafe {
            send_message_low_level(process_id, ptr as i32, len as i32);
        }
    }
    fn set_get_state(&mut self, ptr: u32, len: u32) {
        unsafe { set_get_state(ptr, len) };
    }

    fn provide_value<T: Serialize>(&self, s: &str, val: T) {

        let val = serde_json::to_string(&val).unwrap();
        let val = val.as_bytes();
        // TODO: I need to fix the lifetime of this thing
        // most likely
        unsafe {
            provide_value_low_level(
                s.as_ptr() as i32,
                s.len() as i32,
                val.as_ptr() as i32,
                val.len() as i32,
            );
        }
    }

    fn send_event(&self, kind: &str, event: String) {
        unsafe {
            send_event_low_level(
                kind.as_ptr() as i32,
                kind.len() as i32,
                event.as_ptr() as i32,
                event.len() as i32,
            );
        }
    }
    fn subscribe(&self, kind: &str) {
        unsafe {
            subscribe_low_level(kind.as_ptr() as i32, kind.len() as i32);
        }
    }

    fn unsubscribe(&self, kind: String) {
        unsafe {
            unsubscribe_low_level(kind.as_ptr() as i32, kind.len() as i32);
        }
    }

    fn set_cursor_icon(&self, icon: CursorIcon) {
        unsafe {
            set_cursor_icon_low_level(icon as u32);
        }
    }

    fn receive_last_message(&mut self, process_id: i32) -> String {
        let buffer;
        unsafe {
            let (ptr, len) = receive_last_message_low_level(process_id);
            buffer = String::from_raw_parts(ptr as *mut u8, len as usize, len as usize);
        }
        buffer
    }

    fn get_value<T: for<'a> Deserialize<'a>>(&self, name: &str) -> Option<T> {
        let ptr = unsafe { get_value(name.as_ptr() as i32, name.len() as i32) };
        let string = &fetch_string(ptr);
        serde_json::from_str(string).ok()
    }

    fn try_get_value<T: for<'a> Deserialize<'a>>(&self, name: &str) -> Option<T> {
        let ptr = unsafe { try_get_value(name.as_ptr() as i32, name.len() as i32) };
        if ptr == 0 {
            return None;
        }
        let str = &fetch_string(ptr);
        serde_json::from_str(str).ok()
    }
}


pub static mut APPS: Lazy<Vec<Box<dyn App>>> = Lazy::new(Vec::new);

pub static mut CURRENT_APP: usize = 0;
pub static mut CURRENT_EXTERNAL_ID: Option<usize> = None;



#[derive(Debug, Clone)]
pub struct Context {
    pub external_id: Option<usize>,
    pub meta: WidgetMeta,
    pub app_index: usize,
}

pub static mut CONTEXT: Lazy<HashMap<Option<usize>, Context>> = Lazy::new(|| HashMap::new());


#[no_mangle]
pub extern "C" fn set_widget_identifier(identifier: u32) {
    unsafe {
        let app_index = CONTEXT.get(&Some(identifier as usize)).unwrap().app_index;
        CURRENT_APP = app_index;
        CURRENT_EXTERNAL_ID = Some(identifier as usize);
    }
}

#[no_mangle]
pub extern "C" fn clear_widget_identifier() {
    unsafe {
        CURRENT_APP = 0;
        CURRENT_EXTERNAL_ID = None;
    }
}

#[no_mangle]
pub extern "C" fn on_click(x: f32, y: f32) {
    let app = get_app!();
    unsafe { app.on_click(x, y) }
}

#[no_mangle]
pub extern "C" fn on_delete() {
    println!("Deleting 6");
    let app = get_app!();
    unsafe { app.on_delete() }
}
#[no_mangle]
pub extern "C" fn on_mouse_down(x: f32, y: f32) {
    let app = get_app!();
    unsafe { app.on_mouse_down(x, y) }
}

#[no_mangle]
pub extern "C" fn on_mouse_up(x: f32, y: f32) {
    let app = get_app!();
    unsafe { app.on_mouse_up(x, y) }
}

#[no_mangle]
pub extern "C" fn on_process_message(process_id: u32, str_ptr: u32) {
    let app = get_app!();
    let message = fetch_string(str_ptr);
    unsafe { app.on_process_message(process_id, message) }
}

#[no_mangle]
pub extern "C" fn draw_debug() {
    let debug = unsafe { &DEBUG };
    if debug.is_empty() {
        return;
    }
    let foreground = Color::parse_hex("#62b4a6");
    let background = Color::parse_hex("#1c041e");
    let mut canvas = Canvas::new();
    canvas.set_color(&background);
    canvas.draw_rrect(Rect::new(0.0, 0.0, 300.0, 300.0), 20.0);
    canvas.set_color(&foreground);
    canvas.translate(0.0, 30.0);
    canvas.draw_str(&format!("Debug {}", unsafe { &DEBUG }.len()), 0.0, 0.0);
    canvas.translate(0.0, 30.0);
    for line in unsafe { &DEBUG } {
        canvas.draw_str(line, 0.0, 0.0);
        canvas.translate(0.0, 30.0);
    }
}

#[no_mangle]
pub extern "C" fn on_key(key: u32, state: u32, modifiers: u32) {
    let app = get_app!();
    unsafe { app.on_key(KeyboardInput::from_u32(key, state, modifiers)) }
}

#[no_mangle]
pub extern "C" fn on_mouse_move(x: f32, y: f32, x_diff: f32, y_diff: f32) {
    let app = get_app!();
    unsafe { app.on_mouse_move(x, y, x_diff, y_diff) }
}

#[no_mangle]
pub extern "C" fn on_event(kind_ptr: u32, event_ptr: u32) {
    let app = get_app!();
    let kind = fetch_string(kind_ptr);
    let event = fetch_string(event_ptr);
    unsafe { app.on_event(kind, event) }
}

#[no_mangle]
pub extern "C" fn on_size_change(width: f32, height: f32) {
    let app = get_app!();
    let context = unsafe { CONTEXT.get_mut(&CURRENT_EXTERNAL_ID) }.unwrap();
    context.meta.size.width = width;
    context.meta.size.height = height;
    unsafe { app.on_size_change(width, height) }
}

#[no_mangle]
pub extern "C" fn on_move(x: f32, y: f32) {
    let app = get_app!();
    let context = unsafe { CONTEXT.get_mut(&CURRENT_EXTERNAL_ID) }.unwrap();
    context.meta.position.x = x;
    context.meta.position.y = y;
    unsafe { app.on_move(x, y) }
}

#[no_mangle]
pub extern "C" fn on_scroll(x: f64, y: f64) {
    let app = get_app!();
    unsafe { app.on_scroll(x, y) }
}

#[no_mangle]
pub extern "C" fn get_state() {
    let app = get_app!();
    let s: String = app.get_state();
    let mut s = s.into_bytes();
    let ptr = s.as_mut_ptr() as usize;
    let len = s.len();
    std::mem::forget(s);

    unsafe { app.set_get_state(ptr as u32, len as u32) };
}

#[no_mangle]
pub extern "C" fn finish_get_state(ptr: usize, len: usize) {
    // Deallocates get_state string
    unsafe {
        let _s = Vec::from_raw_parts(ptr as *mut u8, len, len);
    }
}

#[no_mangle]
pub extern "C" fn set_state(ptr: u32, size: u32) {
    let app = get_app!();
    let data = unsafe { Vec::from_raw_parts(ptr as *mut u8, size as usize, size as usize) };
    let s = String::from_utf8(data);
    match s {
        Ok(s) => {
            // Not quite the same logic here
            // I need set_state to signal failure
            let initial_state = app.get_initial_state();
            let new_state = merge_json(Some(s), initial_state);
            app.set_state(new_state);

            // if let Ok(state) = serde_json::from_str(&s) {
            //     let current_state =
            //         serde_json::to_string(unsafe { &app.get_state() })
            //             .ok();
            //     let new_state = merge_json(
            //         Some(s),
            //         current_state.unwrap_or("{}".to_string()),
            //     );
            //     let new_state = serde_json::from_str(&new_state).unwrap();
            //     unsafe { app.set_state(new_state) }
            // } else {
            // let init_state = serde_json::to_string(unsafe {
            //     &$app::init().get_state()
            // })
            // .unwrap();
            // let new_state = merge_json(Some(s), init_state);
            // if let Ok(state) = serde_json::from_str(&new_state) {
            // unsafe { app.set_state(state) }
            // } else {
            //     println!("Failed to parse state even after merging");
            // }
            // }
        }
        Err(err) => {
            println!("error getting state {:?}", err);
        }
    }
}

#[no_mangle]
pub extern "C" fn draw() {
    let app = get_app!();
    unsafe { app.draw() }
}
// TODO: get rid of this
pub use serde_json;

pub mod macros {

    #[macro_export]
    macro_rules! app {
        ($app:ident) => {
            use framework::AppExtensions;
            #[no_mangle]
            #[cfg(feature="main")]
            fn main(widget_id: usize) {
                let mut app = $app::init();
                app.start();
                unsafe {
                    framework::APPS.push(Box::new(app));
                    framework::CONTEXT.insert(None, framework::Context {
                        external_id: None,
                        meta: framework::WidgetMeta::new(
                            framework::Position { x: 0.0, y: 0.0 },
                            framework::Size { width: 0.0, height: 0.0 },
                            1.0,
                            widget_id,
                            "wasm".to_string(),
                        ),
                        app_index: 0,
                    });
                }
            }
        };
    }

    #[macro_export]
    macro_rules! get_app {
        () => {
            unsafe {
                if let Some(app) = APPS.get_mut(CURRENT_APP) {
                    app
                } else {
                    if CURRENT_APP != 0 {
                        // I need to actually delete this
                        // But really I need to make reloading work properly
                        CURRENT_APP = 0;
                    } else {
                        panic!("No app");
                    }
                    APPS.get_mut(CURRENT_APP).unwrap()
                }
            }
        };
    }
}

pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
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

    pub fn with_alpha(&self, arg: f64) -> Color {
        Color::new(self.r, self.g, self.b, arg as f32)
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyCode {
    Key0,
    Key1,
    Key2,
    Key3,
    Key4,
    Key5,
    Key6,
    Key7,
    Key8,
    Key9,
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Equals,
    Minus,
    LeftBracket,
    RightBracket,
    Backslash,
    Semicolon,
    Apostrophe,
    Comma,
    Period,
    Slash,
    Grave,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Escape,
    Space,
    Return,
    LeftArrow,
    RightArrow,
    UpArrow,
    DownArrow,
    BackSpace,
    Tab = 146,
}

impl KeyCode {
    fn from_u32(key: u32) -> Option<Self> {
        match key {
            0 => Some(KeyCode::Key0),
            1 => Some(KeyCode::Key1),
            2 => Some(KeyCode::Key2),
            3 => Some(KeyCode::Key3),
            4 => Some(KeyCode::Key4),
            5 => Some(KeyCode::Key5),
            6 => Some(KeyCode::Key6),
            7 => Some(KeyCode::Key7),
            8 => Some(KeyCode::Key8),
            9 => Some(KeyCode::Key9),
            10 => Some(KeyCode::A),
            11 => Some(KeyCode::B),
            12 => Some(KeyCode::C),
            13 => Some(KeyCode::D),
            14 => Some(KeyCode::E),
            15 => Some(KeyCode::F),
            16 => Some(KeyCode::G),
            17 => Some(KeyCode::H),
            18 => Some(KeyCode::I),
            19 => Some(KeyCode::J),
            20 => Some(KeyCode::K),
            21 => Some(KeyCode::L),
            22 => Some(KeyCode::M),
            23 => Some(KeyCode::N),
            24 => Some(KeyCode::O),
            25 => Some(KeyCode::P),
            26 => Some(KeyCode::Q),
            27 => Some(KeyCode::R),
            28 => Some(KeyCode::S),
            29 => Some(KeyCode::T),
            30 => Some(KeyCode::U),
            31 => Some(KeyCode::V),
            32 => Some(KeyCode::W),
            33 => Some(KeyCode::X),
            34 => Some(KeyCode::Y),
            35 => Some(KeyCode::Z),
            36 => Some(KeyCode::Equals),
            37 => Some(KeyCode::Minus),
            38 => Some(KeyCode::LeftBracket),
            39 => Some(KeyCode::RightBracket),
            40 => Some(KeyCode::Backslash),
            41 => Some(KeyCode::Semicolon),
            42 => Some(KeyCode::Apostrophe),
            43 => Some(KeyCode::Comma),
            44 => Some(KeyCode::Period),
            45 => Some(KeyCode::Slash),
            46 => Some(KeyCode::Grave),
            47 => Some(KeyCode::F1),
            48 => Some(KeyCode::F2),
            49 => Some(KeyCode::F3),
            50 => Some(KeyCode::F4),
            51 => Some(KeyCode::F5),
            52 => Some(KeyCode::F6),
            53 => Some(KeyCode::F7),
            54 => Some(KeyCode::F8),
            55 => Some(KeyCode::F9),
            56 => Some(KeyCode::F10),
            57 => Some(KeyCode::F11),
            58 => Some(KeyCode::F12),
            59 => Some(KeyCode::Escape),
            60 => Some(KeyCode::Space),
            61 => Some(KeyCode::Return),
            62 => Some(KeyCode::LeftArrow),
            63 => Some(KeyCode::RightArrow),
            64 => Some(KeyCode::UpArrow),
            65 => Some(KeyCode::DownArrow),
            66 => Some(KeyCode::BackSpace),
            146 => Some(KeyCode::Tab),
            _ => None,
        }
    }

    #[allow(unused)]
    fn as_u32(&self) -> u32 {
        *self as u32
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KeyState {
    Pressed,
    Released,
}

impl KeyState {
    #[allow(unused)]
    pub fn to_u32(&self) -> u32 {
        match self {
            KeyState::Pressed => 0,
            KeyState::Released => 1,
        }
    }

    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => KeyState::Pressed,
            1 => KeyState::Released,
            _ => panic!("Invalid value for KeyState"),
        }
    }
}

// Not the most efficient representation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub ctrl: bool,
    pub option: bool,
    pub cmd: bool,
}

impl Modifiers {
    pub fn from_u32(modifiers: u32) -> Self {
        Self {
            shift: (modifiers & 1) != 0,
            ctrl: (modifiers & 2) != 0,
            option: (modifiers & 4) != 0,
            cmd: (modifiers & 8) != 0,
        }
    }

    #[allow(unused)]
    pub fn to_u32(&self) -> u32 {
        let mut result = 0;
        if self.shift {
            result |= 1;
        }
        if self.ctrl {
            result |= 2;
        }
        if self.option {
            result |= 4;
        }
        if self.cmd {
            result |= 8;
        }
        result
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyboardInput {
    pub state: KeyState,
    pub key_code: KeyCode,
    pub modifiers: Modifiers,
}

impl KeyboardInput {
    pub fn from_u32(key: u32, state: u32, modifiers: u32) -> Self {
        Self {
            state: KeyState::from_u32(state),
            key_code: KeyCode::from_u32(key).unwrap_or_else(|| panic!("Unknown key code {}", key)),
            modifiers: Modifiers::from_u32(modifiers),
        }
    }

    #[allow(unused)]
    pub fn to_u32_tuple(&self) -> (u32, u32, u32) {
        (
            self.key_code.as_u32(),
            self.state.to_u32(),
            self.modifiers.to_u32(),
        )
    }

    #[allow(unused)]
    pub fn to_char(&self) -> Option<char> {
        match (self.key_code, self.modifiers.shift) {
            (KeyCode::Key0, false) => Some('0'),
            (KeyCode::Key1, false) => Some('1'),
            (KeyCode::Key2, false) => Some('2'),
            (KeyCode::Key3, false) => Some('3'),
            (KeyCode::Key4, false) => Some('4'),
            (KeyCode::Key5, false) => Some('5'),
            (KeyCode::Key6, false) => Some('6'),
            (KeyCode::Key7, false) => Some('7'),
            (KeyCode::Key8, false) => Some('8'),
            (KeyCode::Key9, false) => Some('9'),
            (KeyCode::A, false) => Some('a'),
            (KeyCode::B, false) => Some('b'),
            (KeyCode::C, false) => Some('c'),
            (KeyCode::D, false) => Some('d'),
            (KeyCode::E, false) => Some('e'),
            (KeyCode::F, false) => Some('f'),
            (KeyCode::G, false) => Some('g'),
            (KeyCode::H, false) => Some('h'),
            (KeyCode::I, false) => Some('i'),
            (KeyCode::J, false) => Some('j'),
            (KeyCode::K, false) => Some('k'),
            (KeyCode::L, false) => Some('l'),
            (KeyCode::M, false) => Some('m'),
            (KeyCode::N, false) => Some('n'),
            (KeyCode::O, false) => Some('o'),
            (KeyCode::P, false) => Some('p'),
            (KeyCode::Q, false) => Some('q'),
            (KeyCode::R, false) => Some('r'),
            (KeyCode::S, false) => Some('s'),
            (KeyCode::T, false) => Some('t'),
            (KeyCode::U, false) => Some('u'),
            (KeyCode::V, false) => Some('v'),
            (KeyCode::W, false) => Some('w'),
            (KeyCode::X, false) => Some('x'),
            (KeyCode::Y, false) => Some('y'),
            (KeyCode::Z, false) => Some('z'),
            (KeyCode::Key0, true) => Some(')'),
            (KeyCode::Key1, true) => Some('!'),
            (KeyCode::Key2, true) => Some('@'),
            (KeyCode::Key3, true) => Some('#'),
            (KeyCode::Key4, true) => Some('$'),
            (KeyCode::Key5, true) => Some('%'),
            (KeyCode::Key6, true) => Some('^'),
            (KeyCode::Key7, true) => Some('&'),
            (KeyCode::Key8, true) => Some('*'),
            (KeyCode::Key9, true) => Some('('),
            (KeyCode::A, true) => Some('A'),
            (KeyCode::B, true) => Some('B'),
            (KeyCode::C, true) => Some('C'),
            (KeyCode::D, true) => Some('D'),
            (KeyCode::E, true) => Some('E'),
            (KeyCode::F, true) => Some('F'),
            (KeyCode::G, true) => Some('G'),
            (KeyCode::H, true) => Some('H'),
            (KeyCode::I, true) => Some('I'),
            (KeyCode::J, true) => Some('J'),
            (KeyCode::K, true) => Some('K'),
            (KeyCode::L, true) => Some('L'),
            (KeyCode::M, true) => Some('M'),
            (KeyCode::N, true) => Some('N'),
            (KeyCode::O, true) => Some('O'),
            (KeyCode::P, true) => Some('P'),
            (KeyCode::Q, true) => Some('Q'),
            (KeyCode::R, true) => Some('R'),
            (KeyCode::S, true) => Some('S'),
            (KeyCode::T, true) => Some('T'),
            (KeyCode::U, true) => Some('U'),
            (KeyCode::V, true) => Some('V'),
            (KeyCode::W, true) => Some('W'),
            (KeyCode::X, true) => Some('X'),
            (KeyCode::Y, true) => Some('Y'),
            (KeyCode::Z, true) => Some('Z'),
            (KeyCode::Equals, false) => Some('='),
            (KeyCode::Minus, false) => Some('-'),
            (KeyCode::LeftBracket, false) => Some('['),
            (KeyCode::RightBracket, false) => Some(']'),
            (KeyCode::Backslash, false) => Some('\\'),
            (KeyCode::Semicolon, false) => Some(';'),
            (KeyCode::Apostrophe, false) => Some('\''),
            (KeyCode::Comma, false) => Some(','),
            (KeyCode::Period, false) => Some('.'),
            (KeyCode::Slash, false) => Some('/'),
            (KeyCode::Grave, false) => Some('`'),
            (KeyCode::F1, false) => None,
            (KeyCode::F2, false) => None,
            (KeyCode::F3, false) => None,
            (KeyCode::F4, false) => None,
            (KeyCode::F5, false) => None,
            (KeyCode::F6, false) => None,
            (KeyCode::F7, false) => None,
            (KeyCode::F8, false) => None,
            (KeyCode::F9, false) => None,
            (KeyCode::F10, false) => None,
            (KeyCode::F11, false) => None,
            (KeyCode::F12, false) => None,
            (KeyCode::Escape, false) => None,
            (KeyCode::Equals, true) => Some('+'),
            (KeyCode::Minus, true) => Some('_'),
            (KeyCode::LeftBracket, true) => Some('{'),
            (KeyCode::RightBracket, true) => Some('}'),
            (KeyCode::Backslash, true) => Some('|'),
            (KeyCode::Semicolon, true) => Some(':'),
            (KeyCode::Apostrophe, true) => Some('\"'),
            (KeyCode::Comma, true) => Some('<'),
            (KeyCode::Period, true) => Some('>'),
            (KeyCode::Slash, true) => Some('?'),
            (KeyCode::Grave, true) => Some('~'),
            (KeyCode::Space, _) => Some(' '),
            (KeyCode::Return, _) => Some('\n'),
            _ => None,
        }
    }
}
