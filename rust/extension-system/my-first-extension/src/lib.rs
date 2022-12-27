#[no_mangle]
pub extern "C" fn answer(s: &MyStruct) -> i32 {
    println!("a {} b {}", s.a, s.b);
    (unsafe { foo(*s) }) + 42
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MyStruct {
    pub a: i32,
    pub b: i32,
}


#[link(wasm_import_module = "host")]
extern "C" {
    // imports the name `foo` from `the-wasm-import-module`
    fn foo(s: MyStruct) -> i32;
}
