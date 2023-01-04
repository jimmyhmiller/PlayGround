// #[no_mangle]
// pub extern "C" fn answer(s: &MyStruct) -> i32 {
//     println!("a {} b {}", s.a, s.b);
//     (unsafe { foo(*s) }) + 42
// }

// #[repr(C)]
// #[derive(Copy, Clone)]
// pub struct MyStruct {
//     pub a: i32,
//     pub b: i32,
// }


// #[link(wasm_import_module = "host")]
// extern "C" {
//     fn foo(s: MyStruct) -> i32;
// }

static mut STATE: usize = 0;

#[link(wasm_import_module = "host")]
extern "C" {
    // imports the name `foo` from `the-wasm-import-module`
    fn draw_rect(x: f32, y: f32, width: f32, height: f32);
}

#[link(wasm_import_module = "host")]
extern "C" {
    #[link_name = "draw_str"]
    fn draw_str_low_level(ptr: i32, len: i32, x: f32, y: f32);
}




fn draw_str(s: &str, x: f32, y: f32) {
    unsafe {
        draw_str_low_level(s.as_ptr() as i32, s.len() as i32, x, y);
    }
}

#[no_mangle]
pub extern "C" fn on_click() {
    unsafe { STATE += 10; }
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

#[no_mangle]
pub extern "C" fn get_state() -> *const PointerLengthString {
    let s = serde_json::to_string(unsafe { &STATE }).unwrap();
    let p : PointerLengthString = s.into();
    &p as *const _
}

#[no_mangle]
pub extern "C" fn set_state(ptr: i32, len: i32) {
    let s = String::from(PointerLengthString { ptr: ptr as usize, len: len as usize });
    let state: usize = serde_json::from_str(&s).unwrap();
    unsafe { STATE = state; }
}

#[no_mangle]
pub extern "C" fn draw() {

    unsafe {
        draw_rect(0.0, 0.0, 200 as f32, 100 as f32);
        draw_str(&format!("Count: {}", STATE), 40.0, 50.0);
    }
}
