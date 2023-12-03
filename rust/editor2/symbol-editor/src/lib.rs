use code_editor::CodeEditor;
use framework::{WidgetData, App, KeyboardInput, Position, Size, app, Canvas, Color};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct SymbolEditor {
    data: WidgetData,
    editors: Vec<CodeEditor>,
}


impl App for SymbolEditor {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn draw(&mut self) {

        let external_id = self.get_external_id();
        if external_id == 0 {
            let canvas = Canvas::new();
            let background = Color::parse_hex("#dc9941");
            canvas.set_color(&background);
            canvas.draw_rect(0.0, 0.0, self.data.size.width, self.data.size.height);
        } else {
            self.editors.get_mut(0).unwrap().draw()
        }
       
    }

    fn on_click(&mut self, x: f32, y: f32) {

        if self.get_external_id() != 0 {
            self.editors.get_mut(0).unwrap().on_click(x, y);
            return;
        }
        if !self.editors.is_empty() {
            self.editors.pop();
        }
        let mut editor = CodeEditor::init();
        editor.alive = true;
        editor.file_path = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2/symbol-editor/src/lib.rs".to_string();
        editor.visible_range = (35, 54);
        editor.open_file();
        editor.start();
        let mut data = self.data.clone();
        data.position.x = data.position.x + data.size.width + 50.0;
        self.create_widget_ref(42, data);
        self.editors.push(editor);
    }

    fn on_event(&mut self, kind: String, event: String) {
        if let Some(editor) = self.editors.get_mut(0) {
            editor.on_event(kind, event);
            return;
        }
    }

    fn on_key(&mut self, input: KeyboardInput) {
        let external_id = self.get_external_id();
        if external_id == 0 {

        } else {
            self.editors.get_mut(0).unwrap().on_key(input)
        }
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        if self.get_external_id() != 0 {
            self.editors.get_mut(0).unwrap().on_scroll(x, y);
            return;
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        if self.get_external_id() != 0 {
            self.editors.get_mut(0).unwrap().on_size_change(width, height);
            return;
        }
        self.data.size = Size {
            width,
            height
        }
    }

    fn on_move(&mut self, x: f32, y: f32) {
        if self.get_external_id() != 0 {
            self.editors.get_mut(0).unwrap().on_move(x, y);
            return;
        }
        self.data.position = Position {
            x,
            y,
        }
    }

    fn get_position(&self) -> Position {
        if self.get_external_id() != 0 {
            return self.editors.get(0).unwrap().get_position();
        }
        self.data.position
    }

    fn get_size(&self) -> Size {
        if self.get_external_id() != 0 {
            return self.editors.get(0).unwrap().get_size();
        }
        self.data.size
    }

    fn get_initial_state(&self) -> String {
        if self.get_external_id() != 0 {
            return self.editors.get(0).unwrap().get_initial_state();
        }
        serde_json::to_string(&Self {
            data: Default::default(),
            editors: vec![]
        }).unwrap()
       
    }

    fn get_state(&self) -> String {
        if self.get_external_id() != 0 {
            return self.editors.get(0).unwrap().get_state();
        }

        serde_json::to_string(self).unwrap()   
    }

    fn set_state(&mut self, state: String) {

        if self.get_external_id() != 0 {
            self.editors.get_mut(0).unwrap().set_state(state);
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
        }
    }
}


app!(SymbolEditor);