use framework::{App, app, Size, Canvas, Color};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Event {
    kind: String,
    event: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ColorScheme {
    size: Size,
    events: Vec<Event>
}

impl App for ColorScheme {
    type State = Self;

    fn init() -> Self {
        let me = Self {
            size: Size::default(),
            events: Vec::new(),
        };
        me.subscribe("*".to_string());
        me
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        canvas.set_color(&Color::parse_hex("#353f38"));
        canvas.draw_rect(0.0, 0.0, self.size.width, self.size.height);
        canvas.set_color(&Color::parse_hex("#ffffff"));
        
        // We should really by default have a better
        // starting point for text and stuff.

        // I mean actually we probably just need to have a layout manager.
        canvas.translate(0.0, 50.0);

        for event in self.events.iter() {
            // println!("{:?}", event);
            canvas.draw_str(&event.kind, 40.0, 0.0);
            canvas.translate(0.0, 30.0)
        }
    }

    fn on_click(&mut self, x: f32, y: f32) {
        
    }

    fn on_key(&mut self, input: framework::KeyboardInput) {
    }


    // TODO: It would be nice to have scroll work for free
    // by default and only need to deal with overriding it.
    fn on_scroll(&mut self, x: f64, y: f64) {
        
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.size.width = width;
        self.size.height = height;
    }

    fn get_state(&self) -> Self::State {
        self.clone()
    }

    fn set_state(&mut self, state: Self::State) {
        *self = state;
    }

    fn on_event(&mut self, kind: String, event: String) {
        self.events.push(Event { kind, event });
    }

}

app!(ColorScheme);