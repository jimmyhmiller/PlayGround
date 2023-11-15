use framework::{App, Canvas, KeyboardInput, serde_json::{json, self}, app, Color, WidgetData, Position, Size, KeyCode, KeyState, Widget, WidgetMeta};
use serde::{Serialize, Deserialize};


#[derive(Serialize, Deserialize)]
struct MultipleWidgets {
    index: usize,
    widget_data: WidgetData,
    children: Vec<Widget>,
}

#[allow(unused)]
impl App for MultipleWidgets {
    
    fn draw(&mut self) {
        let canvas = Canvas::new();
        let background = Color::parse_hex("#353f38");
        canvas.set_color(&background);
        canvas.draw_rect( 0.0, 0.0, self.widget_data.size.width, self.widget_data.size.height);
        let foreground = Color::parse_hex("#ffffff");
        canvas.set_color(&foreground);
        canvas.draw_str(&format!("index: {}", self.index), 20.0, 40.0);
    }

    fn on_click(&mut self, x: f32, y: f32) {
        let widget_positions: Option<Vec<WidgetMeta>> = self.get_value("widgets");
        if let Some(widget_positions) = widget_positions {
            println!("widget_positions {:?}", widget_positions);
        }
        println!("index {}", self.index);
    }

    fn on_key(&mut self, input: KeyboardInput) {
        if input.state != KeyState::Pressed {
            return;
        }
        match input.key_code {
            KeyCode::C => {
                let mut new_position = self.widget_data.clone();
                new_position.position = Position { 
                    x: self.widget_data.size.width + 20.0 + self.widget_data.position.x, 
                    y: self.widget_data.position.y 
                };
                let widget = self.create_widget(Box::new(Self { index: self.index + 1, widget_data: new_position, children: vec![] }));
                self.children.push(widget);
            }
            KeyCode::S => {
                // TODO: on_move doesn't actually notify 
                // the editor that we moved
                // just changes where we think we are
                // Maybe I don't want to call on_move
                // Maybe I need to think about window manager like
                // functionality

                // let parent_position = self.get_position();
                // for child in self.children.iter_mut() {
                //     let current_position = child.get_position();
                //     println!("Moving! {:?}", current_position);
                //     let mut new_position = current_position.clone();
                //     let mut diff = Position { 
                //         x: parent_position.x - current_position.x,
                //         y: parent_position.y - current_position.y
                //     };
                //     if diff.x.abs() >= 30.0 {
                //         new_position.x -= 1.0;
                //     }
                //     if diff.y.abs() >= 30.0 {
                //         new_position.x -= 1.0
                //     }
                //     child.on_move(diff.x, diff.y)
                // }
            }
            KeyCode::E => {
                // use rand::Rng;
                // let mut rng = rand::thread_rng();
                // for child in self.children.iter_mut() {
                //    let random_x = rng.gen_range(-500..500);
                //    let random_y = rng.gen_range(-500..500);
                //    let mut child_position = child.get_position();
                //    child_position.x += random_x as f32;
                //    child_position.y += random_y as f32;
                //    child.on_move(child_position.x, child_position.y)
                // }
            }
            _ => {}
        }



    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.widget_data.size = Size { width, height };
    }

    fn on_move(&mut self, x: f32, y: f32) {
        self.widget_data.position = Position { x, y };
    }

    fn get_position(&self) -> Position {
        self.widget_data.position
    }

    fn get_size(&self) -> Size {
        self.widget_data.size
    }

    fn get_initial_state(&self) -> String {
        serde_json::to_string(&Self {
            index: 0,
            widget_data: Default::default(),
            children: vec![],
        }).unwrap()
    }

    fn get_state(&self) -> String {
        json!({}).to_string()
    }

    fn set_state(&mut self, state: String) {
        
    }
}

impl MultipleWidgets {
    pub fn init() -> Self {
        Self {
            index: 0,
            widget_data: Default::default(),
            children: vec![],
        }
    }
}

app!(MultipleWidgets);