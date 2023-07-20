use std::collections::HashMap;

use framework::{App, app, Size, Canvas, Color, macros::serde_json};
use lsp_types::SemanticTokensLegend;
use serde::{Serialize, Deserialize};


#[derive(Clone, Debug, Serialize, Deserialize)]
struct ColorScheme {
    size: Size,
    token_legend: Option<SemanticTokensLegend>,
    colors: Vec<String>,
    y_scroll_offset: f32,
    color_mapping: HashMap<String, String>,
}


impl App for ColorScheme {
    type State = Self;

    fn init() -> Self {
        let me = Self {
            size: Size::default(),
            token_legend: None,
            colors: vec![
                "#D36247", "#FFB5A3", "#F58C73", "#B54226", "#D39147", "#FFD4A3", "#F5B873", "#7CAABD",
                "#4C839A", "#33985C", "#83CDA1", "#53B079", "#1C8245",
            ].iter().map(|s| s.to_string()).collect(),
            color_mapping: HashMap::new(),
            y_scroll_offset: 0.0,
        };
        me.subscribe("token_options".to_string());
        me
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();
        canvas.set_color(&Color::parse_hex("#353f38"));
        canvas.draw_rect(0.0, 0.0, self.size.width, self.size.height);
        canvas.set_color(&Color::parse_hex("#ffffff"));

        canvas.translate(0.0, self.y_scroll_offset);

        canvas.translate(0.0, 50.0);

        if let Some(legend) = &self.token_legend {
            for kind in legend.token_types.iter() {
                canvas.translate(0.0, 40.0);
                canvas.draw_str(kind.as_str(), 40.0, 0.0);
                canvas.save();
                canvas.translate(400.0, -20.0);
                for color in self.colors.iter() {
                    canvas.set_color(&Color::parse_hex(color));
                    canvas.draw_rect(0.0, 0.0, 20.0, 20.0);
                    canvas.translate(25.0, 0.0);
                }
                canvas.restore();
            }
        }
    }

    // TODO: I need a nicer way to do deal with clicks.
    // Right now I have to do things to figure out what was clicked
    // on for each UI. I need to do this generically like
    // set on_click handlers, or give UI parts an id or something.
    fn on_click(&mut self, x: f32, y: f32) {
        
    }

    fn on_key(&mut self, input: framework::KeyboardInput) {
        // Need to be able to access clipboard
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
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
        if kind == "token_options" {
            self.token_legend = Some(serde_json::from_str(&event).unwrap());
        }
    }

}

app!(ColorScheme);