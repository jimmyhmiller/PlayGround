use framework::{app, App, Canvas};



struct ProcessSpawner {

}

impl App for ProcessSpawner {
    type State = u32;

    fn init() -> Self {
        ProcessSpawner {  }
    }

    fn draw(&mut self) {
        // needs to be not on draw
        self.provide_f32("radius", (self.get_position().1 / 1000.0) -1.0 );

        let canvas = Canvas::new();
        canvas.draw_rect(0.0, 0.0, 100.0, 100.0);
    }

    fn on_click(&mut self, x: f32, y: f32) {
        self.start_process("my_process".to_string());
    }

    fn on_key(&mut self, input: framework::KeyboardInput) {
        
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        
    }

    fn get_state(&self) -> Self::State {
        0
    }

    fn set_state(&mut self, state: Self::State) {
        
    }
    
}

app!(ProcessSpawner);
