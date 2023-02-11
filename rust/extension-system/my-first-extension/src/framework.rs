use once_cell::sync::Lazy;

#[link(wasm_import_module = "host")]
extern "C" {
    #[link_name = "draw_str"]
    fn draw_str_low_level(ptr: i32, len: i32, x: f32, y: f32);
    fn draw_rect(x: f32, y: f32, width: f32, height: f32);
    fn save();
    fn clip_rect(x: f32, y: f32, width: f32, height: f32);
    fn draw_rrect(x: f32, y: f32, width: f32, height: f32, radius: f32);
    fn translate(x: f32, y: f32);
    fn restore();
}

#[repr(C)]
pub struct PointerLengthString {
    pub ptr: usize,
    pub len: usize,
}

impl From<String> for PointerLengthString {
    fn from(s: String) -> Self {
        Self {
            ptr: s.as_ptr() as usize,
            len: s.len(),
        }
    }
}

impl From<PointerLengthString> for String {
    fn from(s: PointerLengthString) -> Self {
        unsafe { String::from_raw_parts(s.ptr as *mut u8, s.len, s.len) }
    }
}

pub struct Canvas {

}

impl Canvas {
    pub fn new() -> Self {
        Self {

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

    pub fn save(&self) {
        unsafe {
            save();
        }
    }

    pub fn clip_rect(&self, x: f32, y: f32, width: f32, height: f32) {
        unsafe {
            clip_rect(x, y, width, height);
        }
    }

    pub fn draw_rrect(&self, x: f32, y: f32, width: f32, height: f32, radius: f32) {
        unsafe {
            draw_rrect(x, y, width, height, radius);
        }
    }

    pub fn translate(&self, x: f32, y: f32) {
        unsafe {
            translate(x, y);
        }
    }

    pub fn restore(&self) {
        unsafe {
            restore();
        }
    }
}

pub trait App {
    type State;
    fn init() -> Self;
    fn draw(&mut self);
    fn on_click(&mut self);
    fn get_state(&self) -> Self::State;
    fn set_state(&mut self, state: Self::State);
}

mod macros {

    #[macro_export]
    macro_rules! app {
        ($app:ident) => {
            use once_cell::sync::Lazy;
            use crate::framework::{PointerLengthString};
            static mut APP : Lazy<Counter> = Lazy::new(|| $app::init());

            #[no_mangle]
            pub extern "C" fn on_click() {
                unsafe { APP.on_click() }
            }

            #[no_mangle]
            pub extern "C" fn get_state() -> *const PointerLengthString {
                let s = serde_json::to_string(unsafe { &APP.get_state() }).unwrap();
                let p : PointerLengthString = s.into();
                &p as *const _
            }

            #[no_mangle]
            pub extern "C" fn set_state(ptr: i32, len: i32) {
                let s = String::from(PointerLengthString { ptr: ptr as usize, len: len as usize });
                let state: isize = serde_json::from_str(&s).unwrap();
                unsafe { APP.set_state(state)}
            }

            #[no_mangle]
            pub extern "C" fn draw() {
                unsafe { APP.draw() }
            }
        };
    }
}
