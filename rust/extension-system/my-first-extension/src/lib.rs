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
    fn draw_str_low_level(ptr:i32, len: i32, x: f32, y: f32);
}


fn draw_str(s: &str, x: f32, y: f32) {
    unsafe {
        draw_str_low_level(s.as_ptr() as i32, s.len() as i32, x, y);
    }
}

#[no_mangle]
pub extern "C" fn on_click() {
    unsafe { STATE += 1; }
}


#[no_mangle]
pub extern "C" fn draw() {

    unsafe {
        draw_rect(0.0, 0.0, 100 as f32, 100 as f32);
        draw_str(&format!("{}", STATE), 40.0, 50.0);
    }
}
