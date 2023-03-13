use std::fmt::Debug;



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
    fn start_process_low_level(ptr: i32, len: i32) -> i32;
    fn send_message_low_level(process_id: i32, ptr: i32, len: i32);
    #[allow(improper_ctypes)]
    fn recieve_last_message_low_level(process_id: i32) -> (i32, i32);
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
            width: self.width - x * 2.0,
            height: self.height - y * 2.0,
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

    pub fn set_color(&self, color: &Color) {
        unsafe {
            set_color(color.r, color.g, color.b, color.a);
        }
    }
}

pub static mut DEBUG : Vec<String> = Vec::new();

pub trait App {
    type State;
    fn init() -> Self;
    fn draw(&mut self);
    fn on_click(&mut self, x: f32, y: f32);
    fn on_key(&mut self, input: KeyboardInput);
    fn on_scroll(&mut self, x: f64, y: f64);
    fn get_state(&self) -> Self::State;
    fn set_state(&mut self, state: Self::State);
    fn start_process(&mut self, process: String) -> i32 {
        unsafe {
            start_process_low_level(process.as_ptr() as i32, process.len() as i32)
        }
    }
    fn send_message(&mut self, process_id: i32, message: String) {
        unsafe {
            send_message_low_level(process_id, message.as_ptr() as i32, message.len() as i32);
        }
    }
    fn recieve_last_message(&mut self, process_id: i32) -> String {
        let mut buffer = String::new();
        unsafe {
            let (ptr, len) = recieve_last_message_low_level(process_id);
            buffer = String::from_raw_parts(ptr as *mut u8, len as usize, len as usize);
        }
        buffer
    }
    fn add_debug<T: Debug>(&self, name: &str, value: T) {
        unsafe {
            DEBUG.push(format!("{}: {:?}", name, value));
        }
    }
}

#[no_mangle]
pub extern "C" fn alloc_state(size: i32) -> i32 {
    // Still unsure about this.
    // I've had some weird things happen if I don't clone this before
    // use. I feel like I still don't understand
    // what wasmtime actually wants from me.
    let mut buf: Vec<u8> = Vec::with_capacity((size / 8) as usize);
    let ptr = buf.as_mut_ptr();
    std::mem::forget(ptr);
    ptr as i32
}



mod macros {

    #[macro_export]
    macro_rules! app {
        ($app:ident) => {
            use once_cell::sync::Lazy;
            use $crate::framework::{PointerLengthString, KeyboardInput};
            use $crate::framework::DEBUG;
            static mut APP : Lazy<$app> = Lazy::new(|| $app::init());

            #[no_mangle]
            pub extern "C" fn on_click(x: f32, y: f32) {
                unsafe { APP.on_click(x, y) }
            }

            #[no_mangle]
            pub extern "C" fn draw_debug() {
                let debug = unsafe { &DEBUG };
                if debug.len() == 0 {
                    return;
                }
                let foreground = Color::parse_hex("#62b4a6");
                let background = Color::parse_hex("#1c041e");
                let canvas = Canvas::new();
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
                unsafe { APP.on_key(KeyboardInput::from_u32(key, state, modifiers)) }
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
            pub extern "C" fn set_state(ptr: i32, size: i32) {
                let data = unsafe { Vec::from_raw_parts(ptr as *mut u8, size as usize, size as usize) };
                let data = data.clone();
                let s = from_utf8(&data).unwrap();
                if let Ok(state) = serde_json::from_str(&s) {
                    unsafe { APP.set_state(state)}
                } else {
                    println!("set_state: failed to parse state");
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
        if hex.starts_with('#') {
            start = 1;
        }

        let r = i64::from_str_radix(&hex[start..start + 2], 16).unwrap() as f32;
        let g = i64::from_str_radix(&hex[start + 2..start + 4], 16).unwrap() as f32;
        let b = i64::from_str_radix(&hex[start + 4..start + 6], 16).unwrap() as f32;
        Color::new(r / 255.0, g / 255.0, b / 255.0, 1.0)
    }
}


#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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
            _ => None,
        }
    }

    fn to_u32(&self) -> u32 {
        *self as u32
    }

}


#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum KeyState {
    Pressed,
    Released,
}

impl KeyState {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct KeyboardInput {
    pub state: KeyState,
    pub key_code: KeyCode,
    pub modifiers: Modifiers,
}


impl KeyboardInput {
    pub fn from_u32(key: u32, state: u32, modifiers: u32) -> Self {
        Self {
            state: KeyState::from_u32(state),
            key_code: KeyCode::from_u32(key).unwrap(),
            modifiers: Modifiers::from_u32(modifiers),
        }
    }

    pub fn to_u32_tuple(&self) -> (u32, u32, u32) {
        (
            self.key_code.to_u32(),
            self.state.to_u32(),
            self.modifiers.to_u32(),
        )
    }

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
