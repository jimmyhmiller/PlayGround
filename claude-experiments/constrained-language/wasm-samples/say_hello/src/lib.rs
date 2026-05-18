wit_bindgen::generate!({
    world: "say-hello",
    path: "wit",
});

struct Component;

impl Guest for Component {
    fn handle(event: Greeting) {
        let prior = get_greet_counts(&event.name).unwrap_or(0);
        let next = prior + 1;
        put_greet_counts(&event.name, next);
        let text = format!("Hello, {}! (greeted {} time{})", event.name, next, if next == 1 { "" } else { "s" });
        emit_print(&PrintReq { text });
    }
}

export!(Component);
