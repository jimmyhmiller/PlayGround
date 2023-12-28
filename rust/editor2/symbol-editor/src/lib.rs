use std::collections::{HashMap, HashSet};

use code_editor::CodeEditor;
use framework::{app, App, Canvas, Color, KeyboardInput, Position, Rect, Size, WidgetData};
use itertools::Itertools;
use lsp_types::WorkspaceSymbol;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SymbolEditor {
    data: WidgetData,
    editors: HashMap<usize, CodeEditor>,
    symbols: Vec<WorkspaceSymbol>,
    opened: HashSet<String>,
    clicked: bool,
    mouse_location: Option<(f32, f32)>,
}

// TODO: If I want projects, I need to look at cargo.toml files
fn get_project(symbol: &WorkspaceSymbol) -> Option<String> {
    let uri = match &symbol.location {
        lsp_types::OneOf::Left(l) => l.uri.clone(),
        lsp_types::OneOf::Right(l) => l.uri.clone(),
    };
    let path = uri.path();
    let segments = path.split("/").collect::<Vec<&str>>();
    let root = segments.iter().position(|x| *x == "src")?;
    let project = segments.get(root - 1)?;

    return Some(project.to_string());
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

            let size = self.data.size;
            let bounding_rect = Rect::new(0.0, 0.0, size.width, size.height);
            canvas.set_color(&background);
            canvas.clip_rect(bounding_rect);
            canvas.draw_rrect(bounding_rect, 20.0);

            canvas.clip_rect(bounding_rect.with_inset((20.0, 20.0)));

            canvas.set_color(&foreground);
            canvas.save();
            canvas.translate(40.0, 60.0);

            if self.symbols.is_empty() {
                canvas.draw_str("Loading...", 30.0, 30.0);
                canvas.restore();
                return;
            }

            let symbols = self.symbols.clone();
            // TODO: I need to think about how I want the UI for code base exploration
            // to work. We need to be able to see the structure of the code base,
            // but also jump into bits we care about

            // TODO: This is slow
            let projects = symbols
                .iter()
                .sorted_by(|x, y| Ord::cmp(&get_project(x), &get_project(y)))
                .group_by(|x| get_project(x));

            for (project, symbols) in projects.into_iter() {
                if project.is_none() {
                    continue;
                }
                let symbols = symbols.collect_vec();
                let groups = symbols.iter().group_by(|x| &x.container_name);
                let project = project.unwrap();

                if self.mouse_in_bounds(&canvas, -30.0, 200.0, 30.0) {
                    if self.clicked {
                        if self.opened.contains(&project) {
                            self.opened.remove(&project);
                        } else {
                            self.opened.insert(project.clone());
                        }
                    }
                }
                canvas.draw_str(&project, 0.0, 0.0);
                canvas.translate(30.0, 30.0);
                if !self.opened.contains(&project) {
                    canvas.translate(-30.0, 0.0);
                    continue;
                }
                for (group, symbols) in groups.into_iter() {
                    let group = group.clone().unwrap_or("Top Level".to_string());
                    if self.mouse_in_bounds(&canvas, -30.0, 200.0, 30.0) {
                        if self.clicked {
                            if self.opened.contains(&group) {
                                self.opened.remove(&group);
                            } else {
                                self.opened.insert(group.clone());
                            }
                        }
                    }

                    canvas.draw_str(&group, 0.0, 0.0);
                    canvas.translate(30.0, 30.0);
                    if !self.opened.contains(&group) {
                        canvas.translate(-30.0, 0.0);
                        continue;
                    }
                    for symbol in symbols.into_iter() {
                        canvas.draw_str(&symbol.name, 0.0, 0.0);
                        if self.mouse_in_bounds(&canvas, -30.0, 200.0, 30.0) {
                            canvas.draw_rect(0.0, -20.0, 200.0, 30.0);
                            if self.clicked {
                                let mut editor = CodeEditor::init();
                                editor.alive = true;
                                let location = match &symbol.location {
                                    lsp_types::OneOf::Left(l) => l.clone(),
                                    lsp_types::OneOf::Right(_) => panic!("Not supported"),
                                };
                                editor.file_path = location.uri.path().to_string();
                                let start_line = location.range.start.line;
                                let end_line = location.range.end.line + 1;
                                editor.set_visible_range((start_line as usize, end_line as usize));
                                editor.open_file();
                                editor.start();
                                let mut data = self.data.clone();
                                data.position.x = data.position.x + data.size.width + 50.0;
                                data.size = editor.complete_bounds();
                                editor.widget_data = data.clone();
                                let external_id = self.editors.len();
                                self.create_widget_ref(external_id as u32, data);
                                self.editors.insert(external_id, editor);
                            }
                        }
                        canvas.translate(0.0, 30.0);
                    }
                    canvas.translate(-30.0, 30.0);
                }
                canvas.translate(-30.0, 30.0);
            }
            canvas.restore();

            if self.symbols.is_empty() {
                canvas.draw_rect(0.0, 0.0, 30.0, 30.0);
            }
        } else {
            let external_id = self.get_external_id().unwrap();
            self.editors.get_mut(&external_id).unwrap().draw()
        }
        self.clicked = false;
    }

    fn on_click(&mut self, x: f32, y: f32) {
        self.clicked = true;

        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(&external_id).unwrap().on_click(x, y);
            return;
        }
    }

    fn on_delete(&mut self) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.remove(&external_id);
        }
    }

    fn on_event(&mut self, kind: String, event: String) {
        for (_, editor) in self.editors.iter_mut() {
            editor.on_event(kind.clone(), event.clone());
        }

        if kind == "workspace/symbols" {
            // TODO: Handle multiple workspaces
            // Probably need to send project root
            let symbols: Vec<WorkspaceSymbol> = serde_json::from_str(&event).unwrap();
            self.symbols = symbols;
            self.symbols
                .sort_by_key(|x| (get_project(x), x.container_name.clone()));
        }
    }

    fn on_key(&mut self, input: KeyboardInput) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(&external_id).unwrap().on_key(input);
            return;
        }
    }

    fn on_scroll(&mut self, x: f64, y: f64) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(&external_id).unwrap().on_scroll(x, y);
            return;
        }
    }

    fn on_size_change(&mut self, width: f32, height: f32) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors
                .get_mut(&external_id)
                .unwrap()
                .on_size_change(width, height);
            return;
        }
        self.data.size = Size { width, height }
    }

    fn on_mouse_move(&mut self, x: f32, y: f32, x_diff: f32, y_diff: f32) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors
                .get_mut(&external_id)
                .unwrap()
                .on_mouse_move(x, y, x_diff, y_diff);
            return;
        }
        self.mouse_location = Some((x, y));
        self.set_cursor_icon(framework::CursorIcon::Default);
    }

    fn on_move(&mut self, x: f32, y: f32) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(&external_id).unwrap().on_move(x, y);
            return;
        }
        self.data.position = Position { x, y }
    }

    fn get_position(&self) -> Position {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(&external_id).unwrap().get_position();
        }
        self.data.position
    }

    fn get_size(&self) -> Size {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(&external_id).unwrap().get_size();
        }
        self.data.size
    }

    fn get_initial_state(&self) -> String {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(&external_id).unwrap().get_initial_state();
        }
        serde_json::to_string(&Self {
            data: Default::default(),
            editors: HashMap::new(),
            symbols: vec![],
            opened: HashSet::new(),
            clicked: false,
            mouse_location: None,
        })
        .unwrap()
    }

    fn get_state(&self) -> String {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            return self.editors.get(&external_id).unwrap().get_state();
        }

        serde_json::to_string(self).unwrap()
    }

    fn set_state(&mut self, state: String) {
        let external_id = self.get_external_id();
        if let Some(external_id) = external_id {
            self.editors.get_mut(&external_id).unwrap().set_state(state);
            return;
        }
        *self = serde_json::from_str(&state).unwrap();

        // TODO: Move to start when it happens after set_state;
        if self.get_external_ids().is_empty() {
            let mut to_add = Vec::with_capacity(self.editors.len());
            for (external_id, editor) in self.editors.iter_mut() {
                editor.open_file();
                editor.start();
                let data = editor.widget_data.clone();
                to_add.push((*external_id as u32, data));
            }
            for (external_id, data) in to_add.into_iter() {
                self.create_widget_ref(external_id, data);
            }
        }
    }
}

impl SymbolEditor {
    #[allow(dead_code)]
    fn init() -> Self {
        Self {
            data: Default::default(),
            editors: HashMap::new(),
            symbols: vec![],
            opened: HashSet::new(),
            clicked: false,
            mouse_location: None,
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

app!(SymbolEditor);
