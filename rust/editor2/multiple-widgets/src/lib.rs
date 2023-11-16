use framework::{
    app,
    serde_json::{self, json},
    App, Canvas, Color, KeyCode, KeyState, KeyboardInput, Position, Size, Widget, WidgetData,
    WidgetMeta,
};
use lsp_types::SymbolInformation;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct MultipleWidgets {
    index: usize,
    widget_data: WidgetData,
    symbols: Vec<SymbolInformation>,
}

#[derive(Serialize, Deserialize)]
struct SymbolWidget {
    widget_data: WidgetData,
    symbol: SymbolInformation,
}

impl App for SymbolWidget {
    fn draw(&mut self) {
        let canvas = Canvas::new();
        let background = Color::parse_hex("#003f38");
        canvas.set_color(&background);
        canvas.draw_rect(
            0.0,
            0.0,
            self.widget_data.size.width,
            self.widget_data.size.height,
        );
        let foreground = Color::parse_hex("#ffffff");
        canvas.set_color(&foreground);
        canvas.draw_str(&format!("name: {}", self.symbol.name), 20.0, 40.0);
    }

    fn on_click(&mut self, x: f32, y: f32) {}

    fn on_key(&mut self, input: KeyboardInput) {}

    fn on_scroll(&mut self, x: f64, y: f64) {}

    fn on_size_change(&mut self, width: f32, height: f32) {}

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
            widget_data: Default::default(),
            symbol: self.symbol.clone(),
        })
        .unwrap()
    }

    fn get_state(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn set_state(&mut self, state: String) {
        *self = serde_json::from_str(&state).unwrap();
    }
}

fn layout_elements(max_width: f32, elements: Vec<Size>) -> Vec<WidgetData> {
    // We are trying to pack these elements in. They are going to have uniform width, but
    // variable height. We want to pack them in as tightly as possible, but we also want to
    // keep them in order. 
    let starting_data = WidgetData { position: Position { x: 60.0, y: 60.0 }, size: Size { width: 0.0, height: 0.0 } };

    let mut placed_elements: Vec<WidgetData> = Vec::new();
    let element_width = elements[0].width;
    for element in elements.iter() {
        let last_element = placed_elements.last().unwrap_or(&starting_data);
        let mut x = last_element.position.x + last_element.size.width;
        if x + element_width > max_width {
            // We need to wrap to the next line
            x = starting_data.position.x;
        }
        x += 10.0;

        let y = placed_elements.iter()
            .filter(|w| w.position.x == x)
            .map(|w| w.position.y as u32 + w.size.height as u32).max()
            .unwrap_or(0) as f32 + 10.0;

        let widget_data = WidgetData { position: Position { x, y }, size: element.clone() };
        placed_elements.push(widget_data);

    }


    placed_elements



}



#[allow(unused)]
impl App for MultipleWidgets {
    fn start(&mut self) {
        self.subscribe("workspace/symbols")
    }

    fn on_event(&mut self, kind: String, event: String) {
        let symbols: Vec<SymbolInformation> = serde_json::from_str(&event).unwrap();
        self.symbols = symbols;
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        let background = Color::parse_hex("#353f38");
        canvas.set_color(&background);
        canvas.draw_rect(
            0.0,
            0.0,
            self.widget_data.size.width,
            self.widget_data.size.height,
        );
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

                let elements: Vec<Size> = self.symbols.iter().map(|symbol| {
                    Size {
                        width: 500.0,
                        height: ((symbol.location.range.end.line - symbol.location.range.start.line)
                            as f32
                            * 3.0).max(50.0),
                    }
                }).collect();

                let layout = layout_elements(3000.0, elements);
                
                for (symbol, layout) in self.symbols.clone().iter().zip(layout.iter()) {
                    self.create_widget(
                        Box::new(SymbolWidget {
                            widget_data: layout.clone(),
                            symbol: symbol.clone(),
                        }),
                        layout.clone(),
                    );
                }
            }
            _ => {}
        }
    }

    fn on_scroll(&mut self, x: f64, y: f64) {}

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
            symbols: Vec::new(),
        })
        .unwrap()
    }

    fn get_state(&self) -> String {
        serde_json::to_string(self).unwrap()
    }

    fn set_state(&mut self, state: String) {
        *self = serde_json::from_str(&state).unwrap();
    }
}

impl MultipleWidgets {
    pub fn init() -> Self {
        Self {
            index: 0,
            widget_data: Default::default(),
            symbols: Vec::new(),
        }
    }
}

app!(MultipleWidgets);
