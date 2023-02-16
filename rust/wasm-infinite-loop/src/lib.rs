
#[no_mangle]
extern "C" fn loop_forever(label: i32) -> i32 {
    loop {
        println!("Label {}", label);
    }
}
