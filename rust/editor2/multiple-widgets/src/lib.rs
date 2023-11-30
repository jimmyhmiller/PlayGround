use std::{collections::HashMap, cmp::{max, min}};

use framework::{
    app,
    serde_json::{self},
    App, Canvas, Color, KeyCode, KeyState, KeyboardInput, Position, Size, WidgetData, WidgetMeta,
};
use lsp_types::{SymbolInformation, SymbolKind};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct MultipleWidgets {
    index: usize,
    widget_data: WidgetData,
    symbols: Vec<SymbolInformation>,
    #[serde(skip)]
    widget_positions: Vec<WidgetMeta>,
}

#[derive(Serialize, Deserialize)]
struct SymbolWidget {
    widget_data: WidgetData,
    symbol: SymbolInformation,
}

fn symbol_kind_to_num(symbol_kind: SymbolKind) -> usize {
    match symbol_kind {
        SymbolKind::FILE => 1,
        SymbolKind::MODULE => 2,
        SymbolKind::NAMESPACE => 3,
        SymbolKind::PACKAGE => 4,
        SymbolKind::CLASS => 5,
        SymbolKind::METHOD => 6,
        SymbolKind::PROPERTY => 7,
        SymbolKind::FIELD => 8,
        SymbolKind::CONSTRUCTOR => 9,
        SymbolKind::ENUM => 10,
        SymbolKind::INTERFACE => 11,
        SymbolKind::FUNCTION => 12,
        SymbolKind::VARIABLE => 13,
        SymbolKind::CONSTANT => 14,
        SymbolKind::STRING => 15,
        SymbolKind::NUMBER => 16,
        SymbolKind::BOOLEAN => 17,
        SymbolKind::ARRAY => 18,
        SymbolKind::OBJECT => 19,
        SymbolKind::KEY => 20,
        SymbolKind::NULL => 21,
        SymbolKind::ENUM_MEMBER => 22,
        SymbolKind::STRUCT => 23,
        SymbolKind::EVENT => 24,
        SymbolKind::OPERATOR => 25,
        SymbolKind::TYPE_PARAMETER => 26,
        _ => 0,
    }
}

#[allow(unused)]
impl App for SymbolWidget {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        let background = Color::parse_hex("#003f38");
        canvas.set_color(&background);
        let colors: Option<HashMap<usize, String>> = self.try_get_value("color_mappings");
        // println!("got colors! {:?}", colors);
        if let Some(colors) = colors {
            let color = colors.get(&symbol_kind_to_num(self.symbol.kind));
            if let Some(color) = color {
                canvas.set_color(&Color::parse_hex(color));
            }
        }

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
    let starting_data = WidgetData {
        position: Position { x: 60.0, y: 60.0 },
        size: Size {
            width: 0.0,
            height: 0.0,
        },
    };

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

        let y = placed_elements
            .iter()
            .filter(|w| w.position.x == x)
            .map(|w| w.position.y as u32 + w.size.height as u32)
            .max()
            .unwrap_or(0) as f32
            + 10.0;

        let widget_data = WidgetData {
            position: Position { x, y },
            size: *element,
        };
        placed_elements.push(widget_data);
    }

    placed_elements
}

fn layout_elements2(max_width: f32, elements: &Vec<&WidgetMeta>) -> Vec<WidgetMeta> {

    if elements.len() == 0 {
        return Vec::new();
    }

    let x_margin = 30.0;
    let y_margin = 30.0;

    // We are trying to pack these elements in. They are going to have uniform width, but
    // variable height. We want to pack them in as tightly as possible, but we also want to
    // keep them in order.
    let starting_data = WidgetMeta {
        position: Position { x: 0.0, y: 200.0 },
        size: Size {
            width: 0.0,
            height: 0.0,
        },
        id: 0,
        scale: 1.0,
        kind: "example".to_string(),
    };

    let mut placed_elements: Vec<WidgetMeta> = Vec::new();

    for element in elements.iter() {
        let last_element = placed_elements.last().unwrap_or(&starting_data);
        let mut x = last_element.position.x + last_element.size.width;
        if x + element.size.width > max_width {
            // We need to wrap to the next line
            x = starting_data.position.x;
        }

        x += x_margin;



        let mut y = placed_elements
            .iter()
            .filter(|w| {
                // TODO: Be better
                let a1 = x;
                let a2 = x + element.size.width;
                let b1 = w.position.x;
                let b2 = w.position.x + w.size.width;
                let a1 = a1 as i32;
                let a2 = a2 as i32;
                let b1 = b1 as i32;
                let b2 = b2 as i32;
        
                max(a2, b2) - min(a1, b1) < (a2 - a1) + (b2 - b1)
            })
            .map(|w| w.position.y as u32 + w.size.height as u32)
            .max()
            .unwrap_or(0) as f32
            + y_margin;

        y = y.max(starting_data.position.y);


        let widget_data = WidgetMeta {
            position: Position { x, y },
            size: element.size,
            id: element.id,
            scale: element.scale,
            kind: element.kind.clone(),
        };
        placed_elements.push(widget_data);
    }

    placed_elements
}

#[allow(unused)]
impl App for MultipleWidgets {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

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
        canvas.draw_str(&format!("Rearrange windows"), 20.0, 40.0);
    }

    fn on_click(&mut self, x: f32, y: f32) {
        let widget_positions: Option<Vec<WidgetMeta>> = self.get_value("widgets");
        if let Some(widget_positions) = widget_positions {
            self.widget_positions = widget_positions;
        }
        let new_layout = layout_elements2(
            9000.0,
            &(self
                .widget_positions
                .iter()
                .filter(|x| x.scale == 1.0)
                .filter(|x| x.position != self.widget_data.position)
                .filter(|x| !x.kind.contains("Text"))
                .collect()),
        );
        self.provide_value(
            "widgets",
            serde_json::to_string(&new_layout).unwrap().as_bytes(),
        );
        println!("index {}", self.index);
    }

    fn on_key(&mut self, input: KeyboardInput) {
        if input.state != KeyState::Pressed {
            return;
        }
        if input.key_code == KeyCode::C {
            let elements: Vec<Size> = self
                .symbols
                .iter()
                .map(|symbol| Size {
                    width: 500.0,
                    height: ((symbol.location.range.end.line - symbol.location.range.start.line)
                        as f32
                        * 3.0)
                        .max(50.0),
                })
                .collect();

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
            widget_positions: Vec::new(),
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
            widget_positions: Vec::new(),
        }
    }
}

app!(MultipleWidgets);
