use std::collections::HashMap;

use framework::{
    app, serde_json, App, Canvas, Color, CursorIcon, Position, Rect, WidgetData,
};
use lsp_types::SemanticTokensLegend;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ColorScheme {
    widget_data: WidgetData,
    token_legend: Option<SemanticTokensLegend>,
    colors: Vec<String>,
    y_scroll_offset: f32,
    color_mapping: HashMap<usize, String>,
    mouse_location: Option<(f32, f32)>,
    clicked: bool,
}

impl App for ColorScheme {

    fn draw(&mut self) {
        let mut canvas = Canvas::new();
        let background = Color::parse_hex("#353f38");

        let bounding_rect = Rect::new(
            0.0,
            0.0,
            self.widget_data.size.width,
            self.widget_data.size.height,
        );

        canvas.save();
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);

        canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));

        canvas.set_color(&Color::parse_hex("#ffffff"));

        canvas.translate(0.0, self.y_scroll_offset);

        canvas.translate(0.0, 50.0);

        if let Some(legend) = &self.token_legend {
            for (index, kind) in legend.token_types.iter().enumerate() {
                canvas.translate(0.0, 40.0);
                let white = &"#ffffff".to_string();
                let text_color = self.color_mapping.get(&index).unwrap_or(white).clone();
                canvas.set_color(&Color::parse_hex(&text_color));
                canvas.draw_str(&format!("{} {}", kind.as_str(), index), 40.0, 0.0);
                canvas.save();
                canvas.translate(400.0, -20.0);
                for color in self.colors.iter() {
                    if &text_color == color {
                        canvas.set_color(&Color::parse_hex("#ffffff"));
                        canvas.draw_rect(-1.0, -1.0, 22.0, 22.0);
                    }
                    if self.mouse_in_bounds(&canvas, 20.0, 20.0) {
                        if self.clicked {
                            self.color_mapping.insert(index, color.to_string());
                            self.send_event(
                                "color_mapping_changed",
                                serde_json::to_string(&self.color_mapping).unwrap(),
                            );
                        }
                        canvas.set_color(&Color::parse_hex("#ffffff"));
                        canvas.draw_rect(-1.0, -1.0, 22.0, 22.0);
                    }
                    canvas.set_color(&Color::parse_hex(color));
                    canvas.draw_rect(0.0, 0.0, 20.0, 20.0);
                    canvas.translate(25.0, 0.0);
                }
                canvas.restore();
            }
        }
        self.clicked = false;
    }

    // TODO: I need a nicer way to do deal with clicks.
    // Right now I have to do things to figure out what was clicked
    // on for each UI. I need to do this generically like
    // set on_click handlers, or give UI parts an id or something.
    fn on_click(&mut self, _x: f32, _y: f32) {
        self.clicked = true;
    }
    fn on_mouse_move(&mut self, x: f32, y: f32, _x_diff: f32, _y_diff: f32) {
        self.mouse_location = Some((x, y));
        self.set_cursor_icon(CursorIcon::Default);
    }

    fn on_key(&mut self, _input: framework::KeyboardInput) {
        // Need to be able to access clipboard
    }

    fn on_scroll(&mut self, _x: f64, y: f64) {
        self.y_scroll_offset += y as f32;
        if self.y_scroll_offset > 0.0 {
            self.y_scroll_offset = 0.0;
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.widget_data.size.width = width;
        self.widget_data.size.height = height;
    }

    fn on_move(&mut self, x: f32, y: f32) {
        self.widget_data.position = Position { x, y };
    }

    fn get_state(&self) -> String {
        serde_json::to_string(&self).unwrap()
    }

    fn set_state<'a>(&mut self, state: String) {
        let value = serde_json::from_str(&state).unwrap();
        *self = value;
    }

    fn on_event(&mut self, kind: String, event: String) {
        if kind == "token_options" {
            self.token_legend = Some(serde_json::from_str(&event).unwrap());
        }
    }

    fn get_initial_state(&self) -> String {
        let init = Self::init();
        serde_json::to_string(&init).unwrap()
    }
}

impl ColorScheme {

    fn init() -> Self {
        let me = Self {
            widget_data: WidgetData::default(),
            token_legend: None,
            colors: vec![
                "#D36247", "#FFB5A3", "#F58C73", "#B54226", "#D39147", "#FFD4A3", "#F5B873",
                "#7CAABD", "#4C839A", "#33985C", "#83CDA1", "#53B079", "#1C8245",
            ]
            .iter()
            .map(|s| s.to_string())
            .collect(),
            color_mapping: HashMap::new(),
            y_scroll_offset: 0.0,
            mouse_location: None,
            clicked: false,
        };
        me.subscribe("token_options");
        me
    }

    fn mouse_in_bounds(&self, canvas: &Canvas, width: f32, height: f32) -> bool {
        if let Some((x, y)) = self.mouse_location {
            let canvas_position = canvas.get_current_position();
            if x > canvas_position.0
                && x < canvas_position.0 + width
                && y > canvas_position.1
                && y < canvas_position.1 + height
            {
                return true;
            }
        }
        false
    }
}

app!(ColorScheme);
