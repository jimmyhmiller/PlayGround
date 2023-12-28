use std::collections::{HashMap, HashSet};

use code_editor::CodeEditor;
use framework::{app, App, Canvas, Color, KeyboardInput, Position, Rect, Size, WidgetData};
use lsp_types::WorkspaceSymbol;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct PaneEditor {
    data: WidgetData,
    symbols: Vec<WorkspaceSymbol>,
    opened: HashSet<String>,
    clicked: bool,
    mouse_location: Option<(f32, f32)>,
    widget_ref: usize,
}

impl App for PaneEditor {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn start(&mut self) {
        self.subscribe("workspace/symbols");
    }

    fn draw(&mut self) {
        let canvas = Canvas::new();

        // The main pane controls all drawing
        if self.get_external_id().is_some() {
            return;
        }

        let foreground = Color::parse_hex("#dc9941");
        let background = Color::parse_hex("#353f38");

        let size = self.data.size;
        let bounding_rect = Rect::new(0.0, 0.0, size.width, size.height);
        canvas.set_color(&background);
        canvas.clip_rect(bounding_rect);
        canvas.draw_rrect(bounding_rect, 20.0);

        canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));

        for widget in 0..self.widget_ref {
            // TODO: I would need to know the widget size
            canvas.change_widget(widget as u32);
            canvas.set_color(&background);
            canvas.clip_rect(bounding_rect);
            canvas.draw_rrect(bounding_rect, 20.0);

            canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));
            canvas.set_color(&foreground);
            canvas.draw_str(&format!("Hello World! {}", widget), 100.0, 100.0);
        }
        canvas.default_widget();

        self.clicked = false;
    }

    // TODO: Ideally I just render like react
    // and it creates or removes a component, rather
    // than needing to imperatively do it.
    fn on_click(&mut self, x: f32, y: f32) {
        self.clicked = true;
        self.mouse_location = Some((x, y));
        let widget = self.create_widget_ref(self.widget_ref as u32, self.data.clone());
        self.widget_ref += 1;
    }

    fn on_delete(&mut self) {}

    fn on_event(&mut self, kind: String, event: String) {}

    fn on_key(&mut self, input: KeyboardInput) {}

    fn on_scroll(&mut self, x: f64, y: f64) {}

    fn on_size_change(&mut self, width: f32, height: f32) {
        self.data.size = Size { width, height };
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) {}

    fn on_move(&mut self, x: f32, y: f32) {}

    fn get_position(&self) -> Position {
        self.data.position
    }

    fn get_size(&self) -> Size {
        self.data.size
    }

    fn get_initial_state(&self) -> String {
        serde_json::to_string(&Self {
            data: Default::default(),
            symbols: vec![],
            opened: HashSet::new(),
            clicked: false,
            mouse_location: None,
            widget_ref: 0,
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

impl PaneEditor {
    #[allow(dead_code)]
    fn init() -> Self {
        Self {
            data: Default::default(),
            symbols: vec![],
            opened: HashSet::new(),
            clicked: false,
            mouse_location: None,
            widget_ref: 0,
        }
    }

    fn mouse_in_bounds(&self, canvas: &Canvas, offset: f32, width: f32, height: f32) -> bool {
        if let Some((x, y)) = self.mouse_location {
            let canvas_position = canvas.get_current_position();
            let canvas_position = Position {
                x: canvas_position.0,
                y: canvas_position.1 + offset,
            };
            if x > canvas_position.x
                && x < canvas_position.x + width
                && y > canvas_position.y
                && y < canvas_position.y + height
            {
                return true;
            }
        }
        false
    }
}

app!(PaneEditor);
