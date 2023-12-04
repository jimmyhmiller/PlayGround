use std::collections::HashSet;

use code_editor::CodeEditor;
use framework::{app, App, Canvas, Color, KeyboardInput, Position, Rect, Size, WidgetData};
use itertools::Itertools;
use lsp_types::WorkspaceSymbol;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct SymbolEditor {
    data: WidgetData,
    // TODO: I need these to stick around
    // but right now I don't save those widgets
    #[serde(skip)]
    editors: Vec<CodeEditor>,
    symbols: Vec<WorkspaceSymbol>,
    opened: HashSet<String>,
    clicked: bool,
    mouse_location: Option<(f32, f32)>,
}

fn get_project(symbol: &WorkspaceSymbol) -> Option<String> {
    let uri = match &symbol.location {
        lsp_types::OneOf::Left(l) => l.uri.clone(),
        lsp_types::OneOf::Right(l) => l.uri.clone(),
    };
    let path = uri.path();
    let segments = path.split("/").collect::<Vec<&str>>();
    let root = segments.iter().position(|x| *x == "editor2")?;
    let project = segments.get(root + 1)?;

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

            canvas.draw_rect(0.0, 0.0, self.data.size.width, self.data.size.height);
            canvas.clip_rect(Rect::new(
                0.0,
                0.0,
                self.data.size.width,
                self.data.size.height,
            ));

            canvas.set_color(&foreground);
            canvas.save();
            canvas.translate(20.0, 30.0);

            // TODO: I need to think about how I want the UI for code base exploration
            // to work. We need to be able to see the structure of the code base,
            // but also jump into bits we care about
            let projects = self.symbols
                .iter()
                .group_by(|x| get_project(x));
            
            for (project, symbols) in projects.into_iter() {
                if project.is_none() {
                    continue;
                }
                let symbols = symbols.collect_vec();
                let groups = symbols.iter().group_by(|x| &x.container_name);
                let project = project.unwrap();

                // TODO: Fix the bounds issue
                canvas.save();
                canvas.translate(0.0, -30.0);
                if self.mouse_in_bounds(&canvas, 200.0, 30.0) {
                    if self.clicked {
                        if self.opened.contains(&project) {
                            self.opened.remove(&project);
                        } else {
                            self.opened.insert(project.clone());
                        }
                    }
                }
                canvas.restore();
                canvas.draw_str(&project, 0.0, 0.0);
                canvas.translate(30.0, 30.0);
                if !self.opened.contains(&project) {
                    canvas.translate(-30.0, 0.0);
                    continue;
                }
                for (group, symbols) in groups.into_iter() {
                    if group.is_none() {
                        continue;
                    }
                    let group = group.clone().unwrap();
                    canvas.save();
                    canvas.translate(0.0, -30.0);
                    if self.mouse_in_bounds(&canvas, 200.0, 30.0) {
                        if self.clicked {
                            if self.opened.contains(&group) {
                                self.opened.remove(&group);
                            } else {
                                self.opened.insert(group.clone());
                            }
                        }
                    }
                    canvas.restore();

                    canvas.draw_str(&group, 0.0, 0.0);
                    canvas.translate(30.0, 30.0);
                    if !self.opened.contains(&group) {
                        canvas.translate(-30.0, 0.0);
                        continue;
                    }
                    for symbol in symbols.into_iter() {
                        canvas.draw_str(&symbol.name, 0.0, 0.0);
                        canvas.translate(0.0, 30.0);
                    }
                    canvas.translate(-30.0, 30.0);
                }
                canvas.translate(-30.0, 30.0);
            }


            // symbols.dedup_by(|x, y| y.name == x.name);
            // symbols.sort_by(|x, y| Ord::cmp(&x.container_name, &y.container_name));
            // let groups = symbols.iter().group_by(|x| &x.container_name);

            // for (group, symbols) in &groups {
            //     let projects = symbols
            //         .sorted_by(|x, y| Ord::cmp(&get_project(x), &get_project(y)))
            //         .group_by(|x| {
            //             let uri = match &x.location {
            //                 lsp_types::OneOf::Left(l) => l.uri.clone(),
            //                 lsp_types::OneOf::Right(l) => l.uri.clone(),
            //             };
            //             let path = uri.path();
            //             let segments = path.split("/").collect::<Vec<&str>>();
            //             let root = segments.iter().position(|x| *x == "editor2")?;
            //             let project = segments.get(root + 1)?;

            //             return Some(project.to_string());
            //         });

            //     for (project, groups) in projects.into_iter() {
            //         if project.is_none() {
            //             continue;
            //         }
            //         let project = project.unwrap();
            //         canvas.draw_str(&project, 0.0, 0.0);
            //         canvas.translate(0.0, 30.0);
            //         // canvas.translate(0.0, 30.0);
            //         // for group in groups.into_iter() {
            //         //     // if let Some(group) = group {
            //         //         canvas.draw_str(group, 0.0, 0.0);
            //         //         // canvas.translate(30.0, 30.0);
            //         //         // for symbol in symbols {
            //         //         //     canvas.draw_str(&symbol.name, 0.0, 0.0);
            //         //         //     canvas.translate(0.0, 30.0);
            //         //         // }
            //         //         // canvas.translate(-30.0, 30.0);
            //         //     // } else {
            //         //         // canvas.draw_str("Global", 0.0, 0.0);
            //         //         // canvas.translate(30.0, 30.0);
            //         //         // for symbol in symbols {
            //         //         //     if !symbol.name.contains("draw") {
            //         //         //         continue;
            //         //         //     }
            //         //         //     canvas.draw_str(&format!("{:?}", symbol), 0.0, 0.0);
            //         //         //     canvas.translate(0.0, 30.0);
            //         //         // }
            //         //         // canvas.translate(-30.0, 30.0);
            //         //     // }
            //         // }
            //     }

            //     // canvas.draw_str(&symbol.name, 0.0, 0.0);
            //     // canvas.save();
            //     // canvas.translate(0.0, -30.0);
            //     // if self.mouse_in_bounds(&canvas, 200.0, 30.0) {
            //     //     if self.clicked {
            //     //         let mut editor = CodeEditor::init();
            //     //         editor.alive = true;
            //     //         editor.file_path = symbol.location.uri.path().to_string();
            //     //         let start_line = symbol.location.range.start.line;
            //     //         let end_line = symbol.location.range.end.line + 1;
            //     //         editor.visible_range = (start_line as usize, end_line as usize);
            //     //         editor.open_file();
            //     //         editor.start();
            //     //         let mut data = self.data.clone();
            //     //         data.position.x = data.position.x + data.size.width + 50.0;
            //     //         let external_id = self.editors.len() as u32;
            //     //         self.create_widget_ref(external_id, data);
            //     //         self.editors.push(editor);
            //     //     }
            //     // }
            //     // canvas.restore();
            //     // canvas.translate(0.0, 30.0);
            // }
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
            let mut symbols: Vec<WorkspaceSymbol> = serde_json::from_str(&event).unwrap();
            symbols.sort_by_key(|x| (get_project(x), x.container_name.clone()));
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
            self.editors
                .get_mut(external_id)
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
                .get_mut(external_id)
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
            self.editors.get_mut(external_id).unwrap().on_move(x, y);
            return;
        }
        self.data.position = Position { x, y }
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
            opened: HashSet::new(),
            clicked: false,
            mouse_location: None,
        })
        .unwrap()
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
            opened: HashSet::new(),
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
