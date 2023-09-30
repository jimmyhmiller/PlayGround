use std::{collections::HashSet, io::Write};

use framework::CursorIcon;
use nonblock::NonBlockingReader;
use notify::RecursiveMode;
use serde_json::json;

use crate::{
    editor::{self, Editor, PerFrame},
    event::Event,
    keyboard::{KeyCode, KeyboardInput},
    native::open_file_dialog,
    widget::{Position, Size, TextPane, Wasm, Widget, WidgetData},
};

fn into_wini_cursor_icon(cursor_icon: CursorIcon) -> winit::window::CursorIcon {
    match cursor_icon {
        CursorIcon::Default => winit::window::CursorIcon::Default,
        CursorIcon::Text => winit::window::CursorIcon::Text,
    }
}

impl Editor {
    pub fn handle_events(&mut self, events: Vec<Event>) {
        let mut dirty_widgets = HashSet::new();
        for event in events {
            match event {
                Event::DroppedFile { path, x, y } => {
                    if path.extension().unwrap() == "wasm" {
                        let wasm_id = self
                            .wasm_messenger
                            .new_instance(path.to_str().unwrap(), None);
                        self.widget_store.add_widget(Widget {
                            id: 0,
                            on_click: vec![],
                            position: Position { x, y },
                            size: Size {
                                width: 800.0,
                                height: 800.0,
                            },
                            scale: 1.0,
                            ephemeral: false,
                            data: WidgetData::Wasm {
                                wasm: Wasm::new(path.to_str().unwrap().to_string()),
                                wasm_id,
                            },
                        });
                    } else {
                        self.widget_store.add_widget(Widget {
                            id: 0,
                            // TODO: Temp for testing
                            on_click: vec![],
                            position: Position { x, y },
                            scale: 1.0,
                            ephemeral: false,
                            size: Size {
                                width: 800.0,
                                height: 800.0,
                            },
                            data: WidgetData::TextPane {
                                text_pane: TextPane::new(
                                    std::fs::read_to_string(path.clone()).unwrap().into_bytes(),
                                    40.0,
                                ),
                            },
                        });
                    }
                    if let Some(watcher) = &mut self.debounce_watcher {
                        let watcher = watcher.watcher();
                        watcher
                            .watch(path.as_path(), RecursiveMode::NonRecursive)
                            .unwrap();
                    }
                }

                Event::Scroll { x, y } => {
                    let mouse = self.context.mouse_position;
                    for widget in self.widget_store.iter_mut() {
                        if widget.mouse_over(&mouse) {
                            match &mut widget.data {
                                WidgetData::TextPane { text_pane } => {
                                    dirty_widgets.insert(widget.id);
                                    text_pane.on_scroll(x, y, widget.size.height);
                                }
                                WidgetData::Wasm { wasm: _, wasm_id } => {
                                    dirty_widgets.insert(widget.id);
                                    self.wasm_messenger.send_on_scroll(*wasm_id, x, y);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                // TODO: Unused
                Event::MoveWidgetRelative { selector, x, y } => {
                    let widget_ids = selector.select(&self.widget_store);
                    for widget_id in widget_ids {
                        dirty_widgets.insert(widget_id);
                        if let Some(widget) = self.widget_store.get_mut(widget_id) {
                            widget.position.x += x;
                            widget.position.y += y;
                        }
                    }
                }
                Event::MouseMove {
                    x_diff,
                    y_diff,
                    x,
                    y,
                } => {
                    // We are dragging, so don't click
                    // TODO: This isn't really how this should work
                    // We need to let a widget decide or something
                    if x_diff != 0.0 && y_diff != 0.0 {
                        self.context.cancel_click = true;
                    }

                    if self.context.modifiers.ctrl && self.context.modifiers.option {
                        for widget_id in self.selected_widgets.iter() {
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                dirty_widgets.insert(widget.id);
                                widget.size.width += x_diff;
                                widget.size.height += y_diff;
                                widget.on_size_change(
                                    widget.size.width,
                                    widget.size.height,
                                    &mut self.wasm_messenger,
                                );
                            }
                        }
                    } else if self.context.modifiers.ctrl {
                        for widget_id in self.selected_widgets.iter() {
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                dirty_widgets.insert(widget.id);
                                widget.position.x += x_diff;
                                widget.position.y += y_diff;
                                if widget.position.x > self.window.size.width - 300.0 {
                                    widget.scale = 0.1;
                                } else {
                                    widget.scale = 1.0;
                                }
                            }
                        }
                    } else {
                        let mut was_over = false;
                        for widget in self.widget_store.iter() {
                            let position = Position { x, y };
                            let widget_x = position.x - widget.position.x;
                            let widget_y = position.y - widget.position.y;
                            let widget_space = Position {
                                x: widget_x,
                                y: widget_y,
                            };
                            // TODO: I should probably only send this for the top most widget
                            if widget.mouse_over(&position) {
                                was_over = true;
                                match &widget.data {
                                    WidgetData::Wasm { wasm: _, wasm_id } => {
                                        self
                                            .wasm_messenger
                                            .send_on_mouse_move(*wasm_id, &widget_space, x_diff, y_diff);
                                        dirty_widgets.insert(widget.id);
                                    },
                                    _ => {}
                                }
                            }
                        }
                        if !was_over {
                            self.cursor_icon = into_wini_cursor_icon(CursorIcon::Default);
                        }
                    }
                }
                Event::ReloadWidgets => {
                    self.widget_store.clear();
                    self.load_widgets();
                }
                Event::SaveWidgets => {
                    self.save_widgets();
                }
                Event::ReloadWasm(path) => {
                    for widget in self.widget_store.iter_mut() {
                        if let WidgetData::Wasm { wasm, wasm_id } = &mut widget.data {
                            dirty_widgets.insert(widget.id);
                            if path == wasm.path {
                                self.wasm_messenger.send_reload(*wasm_id);
                                // wasm.reload();
                            }
                        }
                    }
                }
                Event::StartProcess(process_id, wasm_id, process_command) => {
                    let mut process = std::process::Command::new(process_command)
                        .stdout(std::process::Stdio::piped())
                        .stdin(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .spawn()
                        .expect("failed to execute process");

                    // grab stdout
                    let stdout = process.stdout.take().unwrap();
                    let stdout = NonBlockingReader::from_fd(stdout).unwrap();
                    let stderr: NonBlockingReader<std::process::ChildStderr> =
                        NonBlockingReader::from_fd(process.stderr.take().unwrap()).unwrap();
                    self.per_frame_actions
                        .push(PerFrame::ProcessOutput { process_id });

                    let parent_widget_id =
                        self.widget_store.get_widget_by_wasm_id(wasm_id).unwrap();

                    let parent = self.widget_store.get(parent_widget_id).unwrap();
                    let mut position = parent.position.offset(parent.size.width + 50.0, 0.0);

                    loop {
                        let mut should_move = false;
                        for widget in self.widget_store.iter() {
                            if widget.position == position {
                                should_move = true;
                            }
                        }
                        if should_move {
                            position.x += 50.0;
                        } else {
                            break;
                        }
                    }

                    let output_widget_id = self.widget_store.add_widget(Widget {
                        id: 0,
                        position,
                        size: Size {
                            width: 800.0,
                            height: 800.0,
                        },
                        scale: 1.0,
                        on_click: vec![],
                        ephemeral: true,
                        data: WidgetData::TextPane {
                            text_pane: TextPane::new(vec![], 40.0),
                        },
                    });

                    self.processes.insert(
                        process_id,
                        editor::Process {
                            process_id,
                            stdout,
                            stdin: process.stdin.take().unwrap(),
                            stderr,
                            output: String::new(),
                            process,
                            output_widget_id,
                            parent_widget_id,
                        },
                    );
                }
                Event::SendProcessMessage(process_id, message) => {
                    if let Some(process) = self.processes.get_mut(&process_id) {
                        // TODO: Handle error
                        process.stdin.write_all(message.as_bytes()).unwrap();
                    }
                }
                Event::ModifiersChanged(modifiers) => {
                    self.context.modifiers = modifiers;
                }
                Event::Event(kind, event) => {
                    // lookup what widgets are listening, call their on_event handler
                    let empty = &HashSet::new();
                    let specific = self.event_listeners.get(&kind).unwrap_or(empty);
                    let all = self.event_listeners.get("*").unwrap_or(empty);

                    let both = specific.union(all);

                    for widget_id in both {
                        if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                            match &mut widget.data {
                                WidgetData::Wasm { wasm: _, wasm_id } => {
                                    dirty_widgets.insert(widget.id);
                                    self.wasm_messenger.send_event(
                                        *wasm_id,
                                        kind.clone(),
                                        event.clone(),
                                    );
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Event::Subscribe(wasm_id, kind) => {
                    let widget_id = self.widget_store.get_widget_by_wasm_id(wasm_id).unwrap();
                    if let Some(listening_widgets) = self.event_listeners.get_mut(&kind) {
                        listening_widgets.insert(widget_id);
                    } else {
                        self.event_listeners
                            .insert(kind, HashSet::from_iter(vec![widget_id]));
                    }
                }
                Event::Unsubscribe(wasm_id, kind) => {
                    let widget_id = self.widget_store.get_widget_by_wasm_id(wasm_id).unwrap();
                    if let Some(listening_widgets) = self.event_listeners.get_mut(&kind) {
                        listening_widgets.retain(|x| *x != widget_id);
                    }
                }
                Event::OpenFile(path) => {
                    // TODO: Handle this better.
                    // Ugly recursive hack and just need to refactor.
                    self.handle_events(vec![Event::Event(
                        "lith/open-file".to_string(),
                        json!({ "path": path }).to_string(),
                    )]);
                }
                Event::KeyEvent {
                    input:
                        KeyboardInput {
                            state: _,
                            key_code,
                            modifiers,
                        },
                } => {
                    if modifiers.cmd && key_code == KeyCode::O {
                        let path = open_file_dialog();
                        if let Some(path) = path {
                            let path = path.replace("file://", "");
                            let code_editor = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2/target/wasm32-wasi/debug/code_editor.wasm";
                            let path_json = json!({ "file_path": path }).to_string();
                            let wasm_id = self
                                .wasm_messenger
                                .new_instance(code_editor, Some(path_json));
                            self.widget_store.add_widget(Widget {
                                id: 0,
                                // TODO: Automatically find an open space
                                // Or make it so you draw it?
                                position: Position { x: 500.0, y: 500.0 },
                                size: Size {
                                    width: 800.0,
                                    height: 800.0,
                                },
                                on_click: vec![],
                                scale: 1.0,
                                data: WidgetData::Wasm {
                                    wasm: Wasm::new(code_editor.to_string()),
                                    wasm_id,
                                },
                                ephemeral: false,
                            });
                            self.events.push(Event::OpenFile(path));
                        }
                    }
                }
                Event::SetCursor(cursor) => match cursor {
                    framework::CursorIcon::Default => {
                        self.cursor_icon = winit::window::CursorIcon::Default;
                    }
                    framework::CursorIcon::Text => {
                        self.cursor_icon = winit::window::CursorIcon::Text;
                    }
                },
                e => {
                    println!("Unhandled event {:?}", e)
                }
            }
        }
        for widget_id in dirty_widgets {
           self.mark_widget_dirty(widget_id)
        }
    }
}
