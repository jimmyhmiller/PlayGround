use framework::{app, App, Canvas};



struct ProcessSpawner {
    state: f32,
}

impl App for ProcessSpawner {
    type State = u32;

    fn init() -> Self {
        ProcessSpawner { 
            state: 0.0,
         }
    }

    fn draw(&mut self) {
        // needs to be not on draw
        self.provide_f32("x", self.get_position().0);
        self.provide_f32("y", self.get_position().1);
        self.provide_f32("z", self.state);

        let canvas = Canvas::new();
        canvas.draw_rect(0.0, 0.0, 100.0, 100.0);
    }

    fn on_click(&mut self, _x: f32, _y: f32) {
        self.start_process("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/echo-test/target/debug/echo-test".to_string());
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {
        
    }

    fn on_scroll(&mut self, _x: f64, y: f64) {
        self.state += y as f32;
    }

    fn get_state(&self) -> Self::State {
        0
    }

    fn set_state(&mut self, _state: Self::State) {
        
    }
    
}

app!(ProcessSpawner);
