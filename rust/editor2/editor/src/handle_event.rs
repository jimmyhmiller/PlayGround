use std::io::Write;

use nonblock::NonBlockingReader;
use notify::RecursiveMode;

use crate::{event::Event, editor::{Editor, PerFrame, self}, widget::{Widget, Position, Size, WidgetData, Wasm, TextPane}};



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
                    for widget_id in self.selected_widgets.iter() {
                        if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                            widget.position.x += x;
                            widget.position.y += y;
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
                Event::StartProcess(process_id, process_command) => {
                    let mut process = std::process::Command::new(process_command)
                        .stdout(std::process::Stdio::piped())
                        .stdin(std::process::Stdio::piped())
                        .stderr(std::process::Stdio::piped())
                        .spawn()
                        .expect("failed to execute process");
                    

                    // grab stdout
                    let stdout = process.stdout.take().unwrap();
                    let non_blocking = NonBlockingReader::from_fd(stdout).unwrap();
                    self.per_frame_actions.push(PerFrame::ProcessOutput {
                        process_id,
                    });

                    let widget_id = self.widget_store.add_widget(Widget {
                        id: 0,
                        position: Position { x: 0.0, y: 0.0 },
                        size: Size {
                            width: 800.0,
                            height: 800.0,
                        },
                        on_click: vec![],
                        data: WidgetData::TextPane {
                            text_pane: TextPane::new(
                                vec![],
                                40.0,
                            ),
                        },
                    });

                    self.processes.insert(process_id, editor::Process {
                        process_id,
                        stdout: non_blocking,
                        stdin: process.stdin.take().unwrap(),
                        stderr: process.stderr.take().unwrap(),
                        output: String::new(),
                        process,
                        widget_id,
                    });

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
                e => {
                    println!("Unhandled event {:?}", e)
                }
            }
        }
    }
}