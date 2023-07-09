use std::{io::Write, collections::HashSet};

use nonblock::NonBlockingReader;
use notify::RecursiveMode;

use crate::{
    editor::{self, Editor, PerFrame},
    event::Event,
    widget::{Position, Size, TextPane, Wasm, Widget, WidgetData},
};

impl Editor {
    pub fn handle_events(&mut self, events: Vec<Event>) {
        for event in events {
            match event {
                Event::DroppedFile { path, x, y } => {
                    if path.extension().unwrap() == "wasm" {
                        let wasm_id = self.wasm_messenger.new_instance(path.to_str().unwrap());
                        self.widget_store.add_widget(Widget {
                            id: 0,
                            on_click: vec![],
                            position: Position { x, y },
                            size: Size {
                                width: 800.0,
                                height: 800.0,
                            },
                            scale: 1.0,
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
                                    text_pane.on_scroll(x, y, widget.size.height);
                                }
                                WidgetData::Wasm { wasm: _, wasm_id } => {
                                    self.wasm_messenger.send_on_scroll(*wasm_id, x, y);
                                }
                                _ => {}
                            }
                        }
                    }
                }
                Event::MoveWidgetRelative { selector, x, y } => {
                    let widget_ids = selector.select(&self.widget_store);
                    for widget_id in widget_ids {
                        if let Some(widget) = self.widget_store.get_mut(widget_id) {
                            widget.position.x += x;
                            widget.position.y += y;
                        }
                    }
                }
                Event::MouseMove {
                    x_diff: x,
                    y_diff: y,
                    ..
                } => {
                    // We are dragging, so don't click
                    if !self.selected_widgets.is_empty() && x != 0.0 && y != 0.0 {
                        self.context.cancel_click = true;
                    }

                    if self.context.modifiers.ctrl && self.context.modifiers.option {
                        for widget_id in self.selected_widgets.iter() {
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                widget.size.width += x;
                                widget.size.height += y;
                                widget.on_size_change(widget.size.width, widget.size.height, &mut self.wasm_messenger);
                            }
                        }
                    } else if self.context.modifiers.ctrl {
                        for widget_id in self.selected_widgets.iter() {
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                widget.position.x += x;
                                widget.position.y += y;
                            }
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
                    let stderr: NonBlockingReader<std::process::ChildStderr> = NonBlockingReader::from_fd(process.stderr.take().unwrap()).unwrap();
                    self.per_frame_actions
                        .push(PerFrame::ProcessOutput { process_id });


                    let parent_widget_id = self.widget_store.get_widget_by_wasm_id(wasm_id).unwrap();

                    let parent = self.widget_store.get(parent_widget_id).unwrap();
                    let position = parent.position.offset(parent.size.width + 50.0, 0.0);

                    let output_widget_id = self.widget_store.add_widget(Widget {
                        id: 0,
                        position,
                        size: Size {
                            width: 800.0,
                            height: 800.0,
                        },
                        scale: 1.0,
                        on_click: vec![],
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
                    if let Some(listening_widgets) = self.event_listeners.get(&kind) {
                        for widget_id in listening_widgets {
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                match &mut widget.data {
                                    WidgetData::Wasm { wasm: _, wasm_id } => {
                                        self.wasm_messenger.send_event(*wasm_id, kind.clone(), event.clone());
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                Event::Subscribe(wasm_id, kind) => {
                    let widget_id = self.widget_store.get_widget_by_wasm_id(wasm_id).unwrap();
                    if let Some(listening_widgets) = self.event_listeners.get_mut(&kind) {
                        listening_widgets.insert(widget_id);
                    } else {
                        self.event_listeners.insert(kind, HashSet::from_iter(vec![widget_id]));
                    }
                }
                Event::Unsubscribe(wasm_id, kind) => {
                    let widget_id = self.widget_store.get_widget_by_wasm_id(wasm_id).unwrap();
                    if let Some(listening_widgets) = self.event_listeners.get_mut(&kind) {
                        listening_widgets.retain(|x| *x != widget_id);
                    }
                }
                e => {
                    println!("Unhandled event {:?}", e)
                }
            }
        }
    }
}
