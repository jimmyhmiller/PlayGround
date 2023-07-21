use framework::{App, app, Size, Canvas, Color, Rect};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Event {
    kind: String,
    event: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EventViewer {
    size: Size,
    events: Vec<Event>,
    y_scroll_offset: f32,
}

impl App for EventViewer {
    type State = Self;

    fn init() -> Self {
        let me = Self {
            size: Size::default(),
            events: Vec::new(),
            y_scroll_offset: 0.0,
        };
        me.subscribe("*".to_string());
        me
    }

    fn draw(&mut self) {
        let mut canvas = Canvas::new();
        let background = Color::parse_hex("#353f38");

        let bounding_rect = Rect::new(
            0.0,
            0.0,
            self.size.width,
            self.size.height,
        );

        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);

        canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));

        canvas.set_color(&Color::parse_hex("#ffffff"));
        
        // We should really by default have a better
        // starting point for text and stuff.
        // I mean actually we probably just need to have a layout manager.
        canvas.translate(0.0, self.y_scroll_offset);

        canvas.translate(0.0, 50.0);

        for event in self.events.iter() {
            // println!("{:?}", event);
            canvas.draw_str(&event.kind, 40.0, 0.0);
            canvas.translate(0.0, 30.0)
        }
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {
    }


    // TODO: It would be nice to have scroll work for free
    // by default and only need to deal with overriding it.
    fn on_scroll(&mut self, _x: f64, y: f64) {
        self.y_scroll_offset += y as f32;
        if self.y_scroll_offset > 0.0 {
            self.y_scroll_offset = 0.0;
        }
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

app!(EventViewer);