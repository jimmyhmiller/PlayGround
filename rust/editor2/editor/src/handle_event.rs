use std::{
    collections::{HashMap, HashSet},
    io::Write,
    str::from_utf8, cell::RefCell, path::Path,
};

use framework::{CursorIcon, Position, Size, WidgetMeta};
use futures::channel::mpsc::channel;
use nonblock::NonBlockingReader;
use notify::RecursiveMode;
use serde_json::json;

use crate::{
    editor::{self, Editor, PerFrame},
    event::Event,
    keyboard::{KeyCode, KeyboardInput, KeyState},
    native::{open_file_dialog, feedback},
    wasm_messenger::{OutMessage, SaveState},
    widget::Widget,
    widget2::{Ephemeral, TextPane, WasmWidget, Widget as _, Image},
};

fn into_wini_cursor_icon(cursor_icon: CursorIcon) -> winit::window::CursorIcon {
    match cursor_icon {
        CursorIcon::Default => winit::window::CursorIcon::Default,
        CursorIcon::Text => winit::window::CursorIcon::Text,
    }
}

impl Editor {
    pub fn handle_events(&mut self, events: Vec<Event>) {
        let mut dirty_widgets: HashSet<(usize, String)> = HashSet::new();
        for event in events {
            match event {
                Event::DroppedFile { path, x, y } => {
                    if path.extension().unwrap() == "wasm" {
                        let next_id = self.widget_store.next_id();
                        let (wasm_id, receiver) = self.wasm_messenger.new_instance(
                            path.to_str().unwrap(),
                            None,
                            self.values.clone(),
                            self.external_sender.as_ref().unwrap().clone(),
                            next_id,
                        );
                        self.widget_store.add_widget(Widget {
                            data: Box::new(WasmWidget {
                                draw_commands: vec![],
                                sender: Some(self.wasm_messenger.get_sender(wasm_id)),
                                receiver: Some(receiver),
                                meta: WidgetMeta::new(
                                    Position { x, y },
                                    Size {
                                        width: 800.0,
                                        height: 800.0,
                                    },
                                    1.0,
                                    next_id,
                                    "WasmWidget".to_string(),
                                ),
                                save_state: SaveState::Unsaved,
                                wasm_non_draw_commands: vec![],
                                external_sender: Some(
                                    self.external_sender.as_ref().unwrap().clone(),
                                ),
                                path: path.to_str().unwrap().to_string(),
                                message_id: 0,
                                pending_messages: HashMap::new(),
                                dirty: true,
                                external_id: None,
                                value_senders: HashMap::new(),
                                atlas: None,
                                offset: Position { x: 0.0, y: 0.0 },
                                size_offset: Size { width: 0.0, height: 0.0 },
                            }),
                        });
                    } else if path.extension().unwrap() == "png" {
                        let next_id = self.widget_store.next_id();
                        self.widget_store.add_widget(Widget {
                            data: Box::new(Image {
                                path: path.to_str().unwrap().to_string(),
                                cache: RefCell::new(None),
                                aspect_ratio: 1.0,
                                meta: WidgetMeta::new(
                                    Position { x, y },
                                    Size {
                                        width: 800.0,
                                        height: 800.0,
                                    },
                                    1.0,
                                    next_id,
                                    "Image".to_string(),
                                ),
                            }),
                        });
                    } else {
                        let next_id = self.widget_store.next_id();
                        self.widget_store.add_widget(Widget {
                            data: Box::new(TextPane::new(
                                std::fs::read_to_string(path.clone()).unwrap().into_bytes(),
                                40.0,
                                WidgetMeta::new(
                                    Position { x, y },
                                    Size {
                                        width: 800.0,
                                        height: 800.0,
                                    },
                                    1.0,
                                    next_id,
                                    "TextPane".to_string(),
                                ),
                            )),
                        });
                    }
                    if let Some(watcher) = &mut self.debounce_watcher {
                        let watcher = watcher.watcher();
                        watcher
                            .watch(path.as_path(), RecursiveMode::NonRecursive)
                            .unwrap();
                    }
                }

                Event::Scroll { x, y, phase: _ } => {
                    // TODO: Need to cancel scroll I got from non-ctrl to ctrl
                    if self.context.modifiers.ctrl && self.context.modifiers.option {
                        self.canvas_scroll_offset.y += y as f32;
                    } else if self.context.modifiers.ctrl {
                        self.canvas_scroll_offset.x -= x as f32;
                        self.canvas_scroll_offset.y += y as f32;
                    } else {
                        let mouse = self.context.mouse_position;
                        for widget in self.widget_store.iter_mut() {
                            if widget.mouse_over(&mouse) {
                                let modified = widget.on_scroll(x, y);
                                if modified {
                                    dirty_widgets.insert((widget.id(), "scroll".to_string()));
                                }
                            }
                        }
                    }
                }
                Event::PinchZoom { delta, phase: _ } => {
                    // TODO: This isn't perfect, but it is passable

                    let min_scale = 0.1;
                    let max_scale = 3.0;

                    if self.canvas_scale == max_scale && delta > 0.0 {
                        continue;
                    }

                    if self.canvas_scale == min_scale && delta < 0.0 {
                        continue;
                    }

                    let screen_x = self.context.mouse_position.x;
                    let screen_y = self.context.mouse_position.y;

                    let scale_change = -delta as f32;
                    let offset_x = -(screen_x * scale_change);
                    let offset_y = -(screen_y * scale_change);

                    let new_scale = self.canvas_scale + delta as f32;
                    let x_with_new_scale = (self.context.raw_mouse_position.x
                        - self.canvas_scroll_offset.x)
                        / new_scale;
                    let y_with_new_scale = (self.context.raw_mouse_position.y
                        - self.canvas_scroll_offset.y)
                        / new_scale;

                    self.canvas_scroll_offset.x -= offset_x;
                    self.canvas_scroll_offset.y -= offset_y;

                    self.canvas_scale += delta as f32;
                    self.canvas_scale = self.canvas_scale.max(min_scale).min(max_scale);
                    self.context.mouse_position = Position {
                        x: x_with_new_scale,
                        y: y_with_new_scale,
                    };
                }
                Event::LeftMouseDown { .. } => {
                    self.context.left_mouse_down = true;
                    self.context.cancel_click = false;
                    self.add_mouse_down();
                }
                Event::LeftMouseUp { .. } => {
                    self.context.left_mouse_down = false;

                    if self.context.modifiers.ctrl || self.context.modifiers.option || self.context.modifiers.shift || self.context.modifiers.cmd {
                        self.context.cancel_click = true;
                    }

                    for moved in self.context.moved.iter() {
                        let widget = self.widget_store.get_mut(*moved).unwrap();
                        self.events.push(Event::Event("widget/moved".to_string(), serde_json::to_string(&widget.meta()).unwrap()));
                    }
                    if !self.context.moved.is_empty() {
                        feedback();
                    }
                    self.context.moved.clear();
                    self.add_mouse_up();
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
                    if x_diff != 0.0 || y_diff != 0.0 {
                        self.context.cancel_click = true;
                    }

                    if self.context.modifiers.ctrl && self.context.modifiers.option {
                        for widget_id in self.selected_widgets.iter() {
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                dirty_widgets.insert((widget.id(), "resize".to_string()));
                                let mut size = widget.size();
                                size.width += x_diff;
                                size.height += y_diff;
                                widget.on_size_change(size.width, size.height);
                            }
                        }
                    } else if self.context.modifiers.ctrl {
                        for widget_id in self.selected_widgets.iter() {
                            let mut moved = vec![];
                            if let Some(widget) = self.widget_store.get_mut(*widget_id) {
                                moved.push(widget.id());
                            }
                            for widget in self.widget_store.iter() {
                                if let Some(parent_id) = widget.parent_id() {
                                    if moved.contains(&parent_id) {
                                        moved.push(widget.id());
                                    }
                                }
                            }
                            self.context.moved.extend(moved.clone());

                            for moved in moved.iter() {
                                let widget = self.widget_store.get_mut(*moved).unwrap();
                                let x = widget.position().x + x_diff;
                                let y = widget.position().y + y_diff;
                                widget.on_move(x, y);
                                self.widget_store.on_move(*moved);
                            }

                        }
                    }
                    let mut was_over = false;
                    for widget in self.widget_store.iter_mut() {
                        let position = Position { x, y };
                        let widget_x = position.x - widget.position().x;
                        let widget_y = position.y - widget.position().y;
                        let widget_space = Position {
                            x: widget_x,
                            y: widget_y,
                        };
                        // TODO: I should probably only send this for the top most widget
                        if widget.mouse_over(&position) {
                            self.active_widget = Some(widget.id());
                            was_over = true;
                            let modified = widget.on_mouse_move(&widget_space, x_diff, y_diff);
                            if modified {
                                dirty_widgets.insert((widget.id(), "mouse_move".to_string()));
                            }
                        }
                    }
                    if !was_over {
                        self.cursor_icon = into_wini_cursor_icon(CursorIcon::Default);
                    }
                }
                Event::ReloadWidgets => {
                    // TODO: This explains weird behavior
                    // I don't actually save the widgets first?
                    // I would need to save, wait and then reload
                    self.widget_store.clear();
                    self.load_widgets();
                }
                Event::SaveWidgets => {
                    self.save_widgets();
                }
                Event::ReloadWasm(path) => {
                    for widget in self.widget_store.iter_mut() {
                        if let Some(widget) = widget.as_wasm_widget_mut() {
                            if path == widget.path {
                                widget.reload().unwrap();
                            }
                        }
                    }
                }
                Event::StartProcess(process_id, widget_id, process_command) => {
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

                    let parent_widget_id = widget_id;

                    let parent = self.widget_store.get(parent_widget_id).unwrap();
                    let mut position = parent.position().offset(parent.size().width + 50.0, 0.0);

                    loop {
                        let mut should_move = false;
                        for widget in self.widget_store.iter() {
                            if widget.position() == position {
                                should_move = true;
                            }
                        }
                        if should_move {
                            position.x += 50.0;
                        } else {
                            break;
                        }
                    }

                    let next_id = self.widget_store.next_id();
                    let data = Box::new(Ephemeral::wrap(Box::new(TextPane::new(
                        vec![],
                        40.0,
                        WidgetMeta::new(
                            position,
                            Size {
                                width: 800.0,
                                height: 800.0,
                            },
                            1.0,
                            next_id,
                            "TextPane".to_string(),
                        ),
                    ))));

                    let output_widget_id = self.widget_store.add_widget(Widget { data });

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
                        // println!("Sending! {}", message);
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
                            let modified = widget.on_event(&kind, &event);
                            if modified {
                                dirty_widgets.insert((widget.id(), "event".to_string()));
                            }
                        }
                    }
                }
                Event::Subscribe(widget_id, kind) => {
                    if let Some(listening_widgets) = self.event_listeners.get_mut(&kind) {
                        listening_widgets.insert(widget_id);
                    } else {
                        self.event_listeners
                            .insert(kind, HashSet::from_iter(vec![widget_id]));
                    }
                }
                Event::Unsubscribe(widget_id, kind) => {
                    if let Some(listening_widgets) = self.event_listeners.get_mut(&kind) {
                        listening_widgets.retain(|x| *x != widget_id);
                    }
                }
                Event::OpenFile(_path) => {
                    // TODO: Handle this better.
                    // Ugly recursive hack and just need to refactor.
                    // self.handle_events(vec![Event::Event(
                    //     "lith/open-file".to_string(),
                    //     json!({ "path": path }).to_string(),
                    // )]);
                }
                Event::KeyEvent {
                    input:
                        input @ KeyboardInput {
                            state,
                            key_code,
                            modifiers,
                        },
                } => {
                    if state == KeyState::Pressed && modifiers.cmd && key_code == KeyCode::Key0 {
                        self.canvas_scale = 1.0;
                        self.canvas_scroll_offset = Position { x: 0.0, y: 0.0 };
                        continue;
                    }

                    if state == KeyState::Pressed &&  modifiers.option && modifiers.ctrl && key_code == KeyCode::D {
                        self.show_debug = !self.show_debug;
                        continue;
                    }

                    if state == KeyState::Pressed && modifiers.cmd && key_code == KeyCode::O {
                        let path = open_file_dialog();
                        if let Some(path) = path {
                            let path = path.replace("file://", "");
                            let code_editor = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2/target/wasm32-wasi/debug/code_editor.wasm";

                            if let Some(watcher) = &mut self.debounce_watcher {
                                let watcher = watcher.watcher();
                                watcher
                                    .watch(Path::new(code_editor), RecursiveMode::NonRecursive)
                                    .unwrap();
                            }
                            let path_json = json!({ "file_path": path }).to_string();
                            let next_id = self.widget_store.next_id();
                            let (wasm_id, receiver) = self.wasm_messenger.new_instance(
                                code_editor,
                                Some(path_json),
                                self.values.clone(),
                                self.external_sender.as_ref().unwrap().clone(),
                                next_id,
                            );
                            let widget_id = self.widget_store.add_widget(Widget {
                                // TODO: Automatically find an open space
                                // Or make it so you draw it?
                                data: Box::new(WasmWidget {
                                    draw_commands: vec![],
                                    sender: Some(self.wasm_messenger.get_sender(wasm_id)),
                                    receiver: Some(receiver),
                                    meta: WidgetMeta::new(
                                        Position { x: 500.0, y: 500.0 },
                                        Size {
                                            width: 800.0,
                                            height: 800.0,
                                        },
                                        1.0,
                                        next_id,
                                        "WasmWidget".to_string(),
                                    ),
                                    save_state: SaveState::Unsaved,
                                    wasm_non_draw_commands: vec![],
                                    external_sender: Some(
                                        self.external_sender.as_ref().unwrap().clone(),
                                    ),
                                    path: code_editor.to_string(),
                                    message_id: 0,
                                    pending_messages: HashMap::new(),
                                    dirty: true,
                                    external_id: None,
                                    value_senders: HashMap::new(),
                                    atlas: None,
                                    offset: Position { x: 0.0, y: 0.0 },
                                    size_offset: Size { width: 0.0, height: 0.0 },
                                }),
                            });
                            self.mark_widget_dirty(widget_id, "open");
                            self.events.push(Event::OpenFile(path));
                        }
                    } else if let Some(widget_id) = self.active_widget {
                        self.mark_widget_dirty(widget_id, "key");
                        if let Some(widget) = self.widget_store.get_mut(widget_id) {
                            widget.on_key(input);
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
                Event::Redraw(widget_id) => self.mark_widget_dirty(widget_id, "redraw"),
                Event::CreateWidget(wasm_id, x, y, width, height, external_id) => {
                    let next_id = self.widget_store.next_id();

                    let (sender, receiver) = channel::<OutMessage>(100000);

                    let widget_id = self.widget_store.add_widget(Widget {
                        data: Box::new(Ephemeral::wrap(Box::new(WasmWidget {
                            draw_commands: vec![],
                            sender: Some(self.wasm_messenger.get_sender(wasm_id as u64)),
                            receiver: Some(receiver),
                            meta: WidgetMeta::new(
                                Position { x, y },
                                Size { width, height },
                                1.0,
                                next_id,
                                "WasmWidget".to_string(),
                            ),
                            save_state: SaveState::Unsaved,
                            wasm_non_draw_commands: vec![],
                            external_sender: Some(self.external_sender.as_ref().unwrap().clone()),
                            path: "TODO".to_string(),
                            message_id: 0,
                            pending_messages: HashMap::new(),
                            dirty: true,
                            external_id: Some(external_id),
                            value_senders: HashMap::new(),
                            atlas: None,
                            offset: Position { x: 0.0, y: 0.0 },
                            size_offset: Size { width: 0.0, height: 0.0 },
                        }))),
                    });
                    self.wasm_messenger.notify_external_sender(
                        wasm_id,
                        external_id,
                        widget_id,
                        sender,
                    )
                }

                // This works. But I might want to do an event on widgets changed instead
                Event::ValueNeeded(name, widget_id) => match name.as_str() {
                    "widgets" => {
                        let widget_positions: Vec<WidgetMeta> = self
                            .widget_store
                            .iter()
                            .map(|w| {
                                WidgetMeta::new(
                                    w.position(),
                                    w.size(),
                                    w.scale(),
                                    w.id(),
                                    w.typetag_name().to_string(),
                                )
                            })
                            .collect();
                        let widget_positions = serde_json::to_string(&widget_positions).unwrap();
                        if let Some(widget) = self.widget_store.get_mut(widget_id) {
                            if let Some(widget) = widget.as_wasm_widget_mut() {
                                widget.send_value(name, widget_positions.as_bytes().to_vec());
                                dirty_widgets.insert((widget.id(), "value needed".to_string()));
                            }
                        }
                    }
                    name => {
                        if let Some(value) = self.values.get(name) {
                            if let Some(widget) = self.widget_store.get_mut(widget_id) {
                                if let Some(widget) = widget.as_wasm_widget_mut() {
                                    widget.send_value(name.to_string(), value.clone());
                                    dirty_widgets.insert((widget.id(), "value needed".to_string()));
                                }
                            }
                        }
                    }
                },

                Event::ValueNeeded2(name, mut sender) => match name.as_str() {
                    "widgets" => {
                        let widget_positions: Vec<WidgetMeta> = self
                            .widget_store
                            .iter()
                            .map(|w| {
                                WidgetMeta::new(
                                    w.position(),
                                    w.size(),
                                    w.scale(),
                                    w.id(),
                                    w.typetag_name().to_string(),
                                )
                            })
                            .collect();
                        // TODO: Need sending widget to be marked dirty
                        // Do wasm instances know what widget they are?
                        // I think not
                        let widget_positions = serde_json::to_string(&widget_positions).unwrap();
                        match sender.try_send(widget_positions.as_bytes().to_vec()) {
                            Ok(_) => {}
                            Err(e) => {
                                println!("Failed to send {:?}", e);
                            }
                        }
                        
                        // if let Some(widget) = self.widget_store.get_mut(widget_id) {
                        //     if let Some(widget) = widget.as_wasm_widget_mut() {
                        //         widget.send_value(name, widget_positions.as_bytes().to_vec());
                        //         dirty_widgets.insert((widget.id(), "value needed".to_string()));
                        //     }
                        // }
                    }
                    name => {
                        if let Some(value) = self.values.get(name) {
                            match sender.try_send(value.to_vec()) {
                                Ok(_) => {}
                                Err(_) => {
                                    println!("Failed to send");
                                }
                            }
                            // if let Some(widget) = self.widget_store.get_mut(widget_id) {
                            //     if let Some(widget) = widget.as_wasm_widget_mut() {
                            //         widget.send_value(name.to_string(), value.clone());
                            //         dirty_widgets.insert((widget.id(), "value needed".to_string()));
                            //     }
                            // }
                        }
                    }
                },

                Event::ProvideValue(name, value) => match name.as_str() {
                    "widgets" => {
                        let widget_positions: Vec<WidgetMeta> =
                            serde_json::from_str(from_utf8(&value).unwrap()).unwrap();
                        for meta in widget_positions {
                            if let Some(widget) = self.widget_store.get_mut(meta.id) {
                                if widget.position() != meta.position {
                                    widget.on_move(meta.position.x, meta.position.y);
                                    dirty_widgets.insert((widget.id(), "provide value".to_string()));
                                }
                                
                                if widget.size() != meta.size {
                                    widget.on_size_change(meta.size.width, meta.size.height);
                                    dirty_widgets.insert((widget.id(), "provide value".to_string()));
                                }

                                if widget.scale() != meta.scale {
                                    widget.set_scale(meta.scale);
                                    dirty_widgets.insert((widget.id(), "provide value".to_string()));
                                }
                                
                                if widget.parent_id() != meta.parent_id {
                                    widget.set_parent_id(meta.parent_id);
                                    dirty_widgets.insert((widget.id(), "provide value".to_string()));
                                }
                            }
                        }
                    }
                    _ => {
                        self.values.insert(name, value);
                    }
                },

                Event::SmartMagnify { x: _, y: _ } => {
                    for widget in self.widget_store.iter_mut() {
                        if widget.mouse_over(&self.context.mouse_position) {
                            self.canvas_scroll_offset.x = -widget.position().x;
                            self.canvas_scroll_offset.y = -widget.position().y;
                            // TODO: Need to set the scale
                        }
                    }
                }

                Event::MarkDirty(widget_id) => {
                    println!("Marking dirty {}", widget_id);
                    self.mark_widget_dirty(widget_id as usize, "mark dirty");
                }

                _e => {
                    // println!("Unhandled event {:?}", e)
                }
            }
        }
        for (widget_id, reason) in dirty_widgets {
            self.mark_widget_dirty(widget_id, &reason);
        }
    }

    fn add_mouse_down(&mut self) {
        self.context.cancel_click = false;
        let mut mouse_over = vec![];
        // I would need some sort of hierarchy here
        // Right now if widgets are in a stack I would say mouse down
        // on all of them, rather than z-order
        // But I don't have a real defined z-order
        // Maybe I should do the first? Or the last?
        // Not sure
        let mut found_a_widget = false;
        for widget in self.widget_store.iter_by_z_index_mut() {
            if widget.mouse_over(&self.context.mouse_position) {
                found_a_widget = true;
                mouse_over.push(widget.id());
                // We are only selecting the widget for the purposes
                // of dragging if ctrl is down
                if self.context.modifiers.ctrl {
                    self.selected_widgets.insert(widget.id());
                } else {
                    widget.on_mouse_down(&self.context.mouse_position);
                }
                // TODO: This is ugly, just setting the active widget
                // over and over again then we will get the last one
                // which would probably draw on top anyways.
                // Should do better
                self.active_widget = Some(widget.id());
            }
            if found_a_widget {
                break;
            }
        }
        if !found_a_widget {
            self.active_widget = None;
        }
        for id in mouse_over {
            self.respond_to_event(Event::WidgetMouseDown { widget_id: id });
        }
    }

    fn add_mouse_up(&mut self) {
        // This is only true now. I could have a selection mode
        // Or it could be click to select. So really not sure
        // what to do here. But for now I just want to be able to move widgets
        self.selected_widgets.clear();

        let mut to_delete = vec![];

        let mut mouse_over = vec![];
        // TODO: Probably only top widget
        for widget in self.widget_store.iter_mut() {
            if widget.mouse_over(&self.context.mouse_position) {
                mouse_over.push(widget.id());

                let modifiers = self.context.modifiers;
                if modifiers.cmd && modifiers.ctrl && modifiers.option {
                    println!("Deleting 1");
                    widget.on_delete();
                    to_delete.push(widget.id());
                    continue;
                }

                if self.context.cancel_click {
                    widget.on_mouse_up(&self.context.mouse_position);
                } else {
                    widget.on_click(&self.context.mouse_position);
                }
            }
        }
        for id in mouse_over {
            self.respond_to_event(Event::WidgetMouseUp { widget_id: id });
        }
        for widget_id in to_delete {
            self.widget_store.delete_widget(widget_id);
        }
        if self.context.cancel_click {
            self.context.cancel_click = false;
        }

    }
}
