

#[link(wasm_import_module = "host")]
extern "C" {
    #[link_name = "draw_str"]
    fn draw_str_low_level(ptr: i32, len: i32, x: f32, y: f32);
    fn draw_rect(x: f32, y: f32, width: f32, height: f32);
    fn set_color(r: f32, g: f32, b: f32, a: f32);
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
            width: self.width - x,
            height: self.height - y,
        }

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

    pub fn set_color(&self, color: Color) {
        unsafe {
            set_color(color.r, color.g, color.b, color.a);
        }
    }
}

pub trait App {
    type State;
    fn init() -> Self;
    fn draw(&mut self);
    fn on_click(&mut self);
    fn on_scroll(&mut self, x: f64, y: f64);
    fn get_state(&self) -> Self::State;
    fn set_state(&mut self, state: Self::State);
}

mod macros {

    #[macro_export]
    macro_rules! app {
        ($app:ident) => {
            use once_cell::sync::Lazy;
            use crate::framework::{PointerLengthString};
            static mut APP : Lazy<$app> = Lazy::new(|| $app::init());

            #[no_mangle]
            pub extern "C" fn on_click() {
                unsafe { APP.on_click() }
            }

            #[no_mangle]
            pub extern "C" fn on_scroll(x: f64, y: f64) {
                unsafe { APP.on_scroll(x, y) }
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
                if let Ok(state) = serde_json::from_str(&s) {
                    unsafe { APP.set_state(state)}
                }
            }

            #[no_mangle]
            pub extern "C" fn draw() {
                unsafe { APP.draw() }
            }
        };
    }
}



pub struct Color {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Color {

    pub fn new(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    pub fn parse_hex(hex: &str) -> Color {
        let mut start = 0;
        if hex.starts_with("#") {
            start = 1;
        }

        let r = i64::from_str_radix(&hex[start..start + 2], 16).unwrap() as f32;
        let g = i64::from_str_radix(&hex[start + 2..start + 4], 16).unwrap() as f32;
        let b = i64::from_str_radix(&hex[start + 4..start + 6], 16).unwrap() as f32;
        return Color::new(r / 255.0, g / 255.0, b / 255.0, 1.0);
    }
}
