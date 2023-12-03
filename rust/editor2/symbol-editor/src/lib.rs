use code_editor::CodeEditor;
use framework::{WidgetData, App, KeyboardInput, Position, Size, app, Canvas, Color, Rect};
use lsp_types::SymbolInformation;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SymbolEditor {
    data: WidgetData,
    // TODO: I need these to stick around
    // but right now I don't save those widgets
    #[serde(skip)]
    editors: Vec<CodeEditor>,
    symbols: Vec<SymbolInformation>,
    clicked: bool,
    mouse_location: Option<(f32, f32)>,
}


impl App for SymbolEditor {
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

        let external_id = self.get_external_id();
        if external_id.is_none() {
            let mut canvas = Canvas::new();
            let foreground = Color::parse_hex("#dc9941");
            let background = Color::parse_hex("#353f38");
            canvas.set_color(&background);

            canvas.draw_rect(0.0, 0.0, self.data.size.width, self.data.size.height);
            canvas.clip_rect(Rect::new(0.0, 0.0, self.data.size.width, self.data.size.height));

            canvas.set_color(&foreground);
            canvas.save();
            canvas.translate(20.0, 30.0);
            for symbol in self.symbols.clone().iter() {
                canvas.draw_str(&symbol.name, 0.0, 0.0);
                canvas.save();
                canvas.translate(0.0, -30.0);
                if self.mouse_in_bounds(&canvas, 200.0, 30.0) {
                    println!("bounds: {}", symbol.name);
                    if self.clicked {
                        println!("clicked!: {}", symbol.name);
                        let mut editor = CodeEditor::init();
                        editor.alive = true;
                        editor.file_path = symbol.location.uri.path().to_string();
                        let start_line = symbol.location.range.start.line;
                        let end_line = symbol.location.range.end.line + 1;
                        editor.visible_range = (start_line as usize, end_line as usize);
                        editor.open_file();
                        editor.start();
                        let mut data = self.data.clone();
                        data.position.x = data.position.x + data.size.width + 50.0;
                        let external_id = self.editors.len() as u32;
                        self.create_widget_ref(external_id, data);
                        self.editors.push(editor);
                    }
                }
                canvas.restore();
                canvas.translate(0.0, 30.0);
            }
            canvas.restore();

            if self.symbols.is_empty() {
                canvas.draw_rect(0.0, 0.0, 30.0, 30.0);
            }

        } else {
            let external_id = self.get_external_id().unwrap();
            self.editors.get_mut(external_id).unwrap().draw()
        }
        self.clicked = false;
    }

    fn on_click(&mut self, x: f32, y: f32) {
        self.clicked = true;

        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().on_click(x, y);
            return;
        }
    }

    fn on_event(&mut self, kind: String, event: String) {

        for editor in self.editors.iter_mut() {
            editor.on_event(kind.clone(), event.clone());
        }

        if kind == "workspace/symbols" {
            let symbols: Vec<SymbolInformation> = serde_json::from_str(&event).unwrap();
            self.symbols = symbols;
        }
    }

    fn on_key(&mut self, input: KeyboardInput) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().on_key(input);
            return;
        }
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().on_scroll(x, y);
            return;
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().on_size_change(width, height);
            return;
        }
        self.data.size = Size {
            width,
            height
        }
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().on_mouse_move(x, y, x_diff, y_diff);
            return;
        }
        self.mouse_location = Some((x, y));
        self.set_cursor_icon(framework::CursorIcon::Default);
    }

    fn on_move(&mut self, x: f32, y: f32) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().on_move(x, y);
            return;
        }
        self.data.position = Position {
            x,
            y,
        }
    }

    fn get_position(&self) -> Position {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(external_id).unwrap().get_position();
        }
        self.data.position
    }

    fn get_size(&self) -> Size {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(external_id).unwrap().get_size();
        }
        self.data.size
    }

    fn get_initial_state(&self) -> String {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(external_id).unwrap().get_initial_state();
        }
        serde_json::to_string(&Self {
            data: Default::default(),
            editors: vec![],
            symbols: vec![],
            clicked: false,
            mouse_location: None,
        }).unwrap()
       
    }

    fn get_state(&self) -> String {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(external_id).unwrap().get_state();
        }

        serde_json::to_string(self).unwrap()   
    }

    fn set_state(&mut self, state: String) {

        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(external_id).unwrap().set_state(state);
            return;
        }
        *self = serde_json::from_str(&state).unwrap();
    }
}

impl SymbolEditor {
    #[allow(dead_code)]
    fn init() -> Self {
        Self {
            data: Default::default(),
            editors: vec![],
            symbols: vec![],
            clicked: false,
            mouse_location: None,
        }
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

app!(SymbolEditor);