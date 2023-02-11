use framework::{App, Canvas};
mod framework;

struct Counter {
    count: isize,
}

impl App for Counter {
    type State = isize;

    fn init() -> Self {
        Self {
            count: 0,
        }
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        canvas.draw_rect(0.0, 0.0, 300 as f32, 100 as f32);
        canvas.draw_str(&format!("Count: {}", self.count), 40.0, 50.0);
    }

    fn on_click(&mut self) {
        self.count += 1;
    }

    fn get_state(&self) -> Self::State {
        self.count
    }

    fn set_state(&mut self, state: Self::State) {
        self.count = state;
    }
}


app!(Counter);

