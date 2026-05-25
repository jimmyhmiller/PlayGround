wit_bindgen::generate!({
    world: "counter",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: u32) {
        let current = get_count();
        set_count(current + event);
    }
}

export!(Component);
