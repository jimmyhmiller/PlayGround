wit_bindgen::generate!({
    world: "bumper",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: u32) {
        let current = get_counter();
        let next = current + event;
        set_counter(next);
        emit_notify(&format!("counter is now {next}"));
    }
}

export!(Component);
