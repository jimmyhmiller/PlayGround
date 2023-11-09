use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ChildStdout,
    sync::mpsc::{Receiver, Sender},
    thread,
    time::Duration,
};

use crate::{
    color::Color,
    event::Event,
    fps_counter::FpsCounter,
    keyboard::Modifiers,
    wasm_messenger::WasmMessenger,
    widget::{Position, Size, Widget, WidgetData, WidgetId, WidgetStore},
    widget2::{TextPane, WasmWidget},
};

use nonblock::NonBlockingReader;
use notify::{FsEventWatcher, RecursiveMode};

use notify_debouncer_mini::{new_debouncer, Debouncer};
use ron::ser::PrettyConfig;
use serde::{Deserialize, Serialize};
use skia_safe::{Canvas, Color4f, Paint, Point};
use skia_safe::{Font, FontStyle, Typeface};
use winit::{event_loop::EventLoopProxy, window::CursorIcon};

pub struct Context {
    pub mouse_position: Position,
    pub left_mouse_down: bool,
    pub right_mouse_down: bool,
    pub cancel_click: bool,
    pub modifiers: Modifiers,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    USize(usize),
    F32(f32),
    String(String),
    Bytes(Vec<u8>),
}

pub enum PerFrame {
    ProcessOutput { process_id: usize },
}

pub struct Process {
    pub process_id: usize,
    pub stdout: NonBlockingReader<ChildStdout>,
    pub stdin: std::process::ChildStdin,
    pub stderr: NonBlockingReader<std::process::ChildStderr>,
    pub output: String,
    // TODO: I could remove this and just allow
    // multiple widgets to be attached to a process.
    pub parent_widget_id: usize,
    pub output_widget_id: usize,
    pub process: std::process::Child,
}

impl Process {
    pub fn kill(&mut self) {
        self.process.kill().unwrap();
    }
}

pub struct Window {
    pub size: Size,
}

pub struct Editor {
    pub events: Events,
    pub fps_counter: FpsCounter,
    pub context: Context,
    pub widget_store: WidgetStore,
    pub should_redraw: bool,
    pub selected_widgets: HashSet<WidgetId>,
    pub active_widget: Option<WidgetId>,
    pub external_receiver: Option<Receiver<Event>>,
    pub external_sender: Option<Sender<Event>>,
    pub debounce_watcher: Option<Debouncer<FsEventWatcher>>,
    pub event_loop_proxy: Option<EventLoopProxy<()>>,
    pub wasm_messenger: WasmMessenger,
    pub widget_config_path: String,
    pub values: HashMap<String, Value>,
    pub processes: HashMap<usize, Process>,
    pub per_frame_actions: Vec<PerFrame>,
    pub event_listeners: HashMap<String, HashSet<WidgetId>>,
    pub window: Window,
    pub cursor_icon: CursorIcon,
    pub dirty_widgets: HashSet<usize>,
    pub first_frame: bool,
}

pub struct Events {
    events: Vec<Event>,
    frame_start_index: usize,
    frame_end_index: usize, // exclusive
}

// It might be unnecessary to have start and end.
// But I like the explicitness for now.
impl Events {
    fn new() -> Events {
        Events {
            events: Vec::new(),
            frame_start_index: 0,
            frame_end_index: 0,
        }
    }

    pub fn push(&mut self, event: Event) {
        self.events.push(event);
    }

    fn push_current_frame(&mut self, event: Event) {
        self.events.push(event);
        self.frame_end_index = self.events.len();
    }

    pub fn events_for_frame(&self) -> &[Event] {
        &self.events[self.frame_start_index..self.frame_end_index]
    }

    fn next_frame(&mut self) {
        self.frame_start_index = self.frame_end_index;
    }

    fn end_frame(&mut self) {
        self.frame_end_index = self.events.len();
    }
}

impl Editor {
    pub fn _set_mouse_position(&mut self, x: f32, y: f32) {
        self.context.mouse_position = Position { x, y };
    }

    pub fn end_frame(&mut self) {
        self.events.end_frame();
        self.should_redraw = false;
    }

    pub fn next_frame(&mut self) {
        self.events.next_frame();
        self.first_frame = false;
    }

    pub fn setup(&mut self) {}

    fn setup_file_watcher(&mut self) {
        let widget_config_path = self.widget_config_path.to_string();
        let (watch_raw_send, watch_raw_receive) = std::sync::mpsc::channel();
        let (sender, receiver) = std::sync::mpsc::channel::<Event>();
        let mut debouncer: Debouncer<FsEventWatcher> =
            new_debouncer(Duration::from_millis(250), None, watch_raw_send).unwrap();

        let watcher = debouncer.watcher();

        // Probably need to debounce this
        watcher
            .watch(Path::new(&widget_config_path), RecursiveMode::NonRecursive)
            .unwrap();

        let sender_clone = sender.clone();

        let event_loop_proxy = self.event_loop_proxy.as_ref().unwrap().clone();
        thread::spawn(move || {
            for res in watch_raw_receive {
                match res {
                    Ok(event) => {
                        event.iter().for_each(|event| {
                            let path = &event.path;
                            if path.to_str().unwrap() == widget_config_path {
                                if sender.send(Event::ReloadWidgets).is_err() {
                                    println!("Failed to reload widgets");
                                }
                            } else if path.extension().unwrap() == "wasm" {
                                sender
                                    .send(Event::ReloadWasm(path.to_str().unwrap().to_string()))
                                    .unwrap();
                            } else {
                                println!("Ignoring event for: {:?}", path);
                            }
                        });
                    }
                    Err(e) => println!("watch error: {:?}", e),
                }
                event_loop_proxy.send_event(()).unwrap();
            }
        });
        self.external_receiver = Some(receiver);
        self.external_sender = Some(sender_clone.clone());
        // self.wasm_messenger.set_external_sender(sender_clone);
        self.debounce_watcher = Some(debouncer);
    }

    pub fn load_widgets(&mut self) {
        let mut file = File::open(&self.widget_config_path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let widgets: Vec<Widget> = contents
            .split(';')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Ok(mut widget) = ron::de::from_str::<Widget>(s) {
                    widget.init(&mut self.wasm_messenger);
                    if let Some(watcher) = &mut self.debounce_watcher {
                        let watcher = watcher.watcher();
                        let files_to_watch = widget.files_to_watch();
                        for path in files_to_watch.iter() {
                            watcher
                                .watch(Path::new(path), RecursiveMode::NonRecursive)
                                .unwrap();
                        }
                    }
                    widget
                } else {
                    println!("Failed to parse: {}", s);
                    panic!("Failed to parse");
                }
            })
            .collect();
        for widget in widgets {
            self.widget_store.add_widget(widget);
        }
        for widget in self.widget_store.iter_mut() {
            if let Some(widget) = widget.data2.as_any_mut().downcast_mut::<WasmWidget>() {
                widget.external_sender = Some(self.external_sender.as_ref().unwrap().clone());
            }
            if widget.position.x >= self.window.size.width
                || widget.position.y >= self.window.size.height
            {
                println!(
                    "Widget out of bounds, moving to edge of screen {}",
                    widget.id
                );
                widget.position.x = widget
                    .position
                    .x
                    .min(self.window.size.width - widget.size.width * widget.scale);
                widget.position.y = widget
                    .position
                    .y
                    .min(self.window.size.height - widget.size.height * widget.scale);
            }
        }
    }

    pub fn update(&mut self) -> bool {
        // TODO: Put in better place
        for widget in self.widget_store.iter_mut() {
            widget.data2.update().unwrap();
        }

        // Todo: Need to test that I am not missing any
        // events with my start and end

        self.wasm_messenger.tick();

        for wasm_id in self.wasm_messenger.get_and_drain_dirty_wasm() {
            if let Some(widget) = self.widget_store.get_mut(wasm_id as usize) {
                self.dirty_widgets.insert(widget.id);
            }
        }

        if let Some(receiver) = &self.external_receiver {
            for event in receiver.try_iter() {
                self.events.push_current_frame(event);
            }
        }

        let events = self.events.events_for_frame().to_vec();
        self.next_frame();

        let events_empty = events.is_empty();

        if !events_empty {
            self.should_redraw = true;
        }

        self.handle_events(events);

        for widget in self.widget_store.iter_mut() {
            widget.data2.update().unwrap();
        }
        !events_empty || self.wasm_messenger.number_of_pending_requests() > 0
    }

    pub fn process_per_frame_actions(&mut self) {
        let mut to_delete = HashSet::new();
        for action in self.per_frame_actions.iter() {
            match action {
                PerFrame::ProcessOutput { process_id } => {
                    if let Some(process) = self.processes.get_mut(process_id) {
                        let stdout = &mut process.stdout;
                        let max_attempts = 100;
                        let mut i = 0;
                        let mut buf = String::new();
                        while !stdout.is_eof() {
                            if i > max_attempts {
                                break;
                            }
                            let length = stdout.read_available_to_string(&mut buf).unwrap();
                            if length == 0 {
                                break;
                            }
                            i += 1;
                        }
                        let stderr = &mut process.stderr;
                        let max_attempts = 100;
                        let mut i = 0;
                        while !stderr.is_eof() {
                            if i > max_attempts {
                                break;
                            }
                            let length = stderr.read_available_to_string(&mut buf).unwrap();
                            if length == 0 {
                                break;
                            }
                            i += 1;
                        }
                        if !buf.is_empty() {
                            process.output.push('\n');
                            process.output.push_str(&buf);
                            if let Some(widget) =
                                self.widget_store.get_mut(process.parent_widget_id)
                            {
                                widget.send_process_message(*process_id, &buf);
                            }
                        }
                    } else {
                        to_delete.insert(*process_id);
                    }
                }
            }

            for process in self.processes.values() {
                let widget_id = process.output_widget_id;
                self.dirty_widgets.insert(widget_id);
                let widget = self.widget_store.get_mut(widget_id).unwrap();
                let output = &process.output;
                match &mut widget.data {
                    WidgetData::TextPane { text_pane } => {
                        text_pane.set_text(output);
                        if let Some(text_pane) =
                            widget.data2.as_any_mut().downcast_mut::<TextPane>()
                        {
                            text_pane.set_text(output);
                        }
                    }
                    WidgetData::Deleted => {
                        to_delete.insert(process.process_id);
                    }
                    _ => unreachable!("Shouldn't be here"),
                }
            }

            for process_id in to_delete.iter() {
                self.processes.remove(process_id);
            }
        }

        self.per_frame_actions.retain(|action| match action {
            PerFrame::ProcessOutput { process_id: id } => !to_delete.contains(id),
        });
    }

    pub fn new() -> Self {
        let mut widget_config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        widget_config_path.push("widgets.ron");
        let widget_config_path = widget_config_path.to_str().unwrap();

        Self {
            events: Events::new(),
            fps_counter: FpsCounter::new(),
            context: Context {
                mouse_position: Position { x: 0.0, y: 0.0 },
                left_mouse_down: false,
                right_mouse_down: false,
                cancel_click: false,
                modifiers: Modifiers::default(),
            },
            widget_store: WidgetStore::new(),
            should_redraw: true,
            selected_widgets: HashSet::new(),
            external_receiver: None,
            external_sender: None,
            debounce_watcher: None,
            event_loop_proxy: None,
            active_widget: None,
            wasm_messenger: WasmMessenger::new(),
            widget_config_path: widget_config_path.to_string(),
            values: HashMap::new(),
            per_frame_actions: Vec::new(),
            processes: HashMap::new(),
            event_listeners: HashMap::new(),
            window: Window {
                size: Size {
                    width: 800.0,
                    height: 800.0,
                },
            },
            cursor_icon: CursorIcon::Default,
            dirty_widgets: HashSet::new(),
            first_frame: true,
        }
    }

    pub fn draw(&mut self, canvas: &Canvas) {
        self.fps_counter.tick();
        use skia_safe::Size;

        let background = Color::parse_hex("#39463e");

        canvas.clear(background.as_color4f());

        let font = Font::new(
            Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(),
            32.0,
        );
        let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);

        // Not drawing fps because it is wrong now that we don't
        // update every frame

        // canvas.draw_str(
        //     self.fps_counter.fps.to_string(),
        //     Point::new(canvas_size.width - 60.0, 30.0),
        //     &font,
        //     white,
        // );

        let canvas_size = Size::from(canvas.base_layer_size());
        canvas.save();
        canvas.translate((canvas_size.width - 300.0, 60.0));
        let counts = self.wasm_messenger.pending_message_counts();
        for line in counts.lines() {
            canvas.draw_str(line, Point::new(0.0, 0.0), &font, white);
            canvas.translate((0.0, 30.0));
        }
        canvas.restore();

        self.widget_store.draw(canvas, &self.dirty_widgets);
        self.dirty_widgets.clear();
    }

    pub fn add_event(&mut self, event: &winit::event::Event<'_, ()>) -> bool {
        if let Some(event) = Event::from_winit_event(event, self.context.modifiers) {
            self.respond_to_event(event);
            return true;
        }
        false
    }

    // TODO: Do I need this indirection?
    pub fn respond_to_event(&mut self, mut event: Event) {
        event.patch_mouse_event(&self.context.mouse_position);
        match event {
            Event::Noop => {}
            Event::MouseMove { x, y, .. } => {
                // I want to be able to respond to mouse move events
                // I just might not want to save them?
                // I'm not sure...
                let x_diff = x - self.context.mouse_position.x;
                let y_diff = y - self.context.mouse_position.y;
                self.events.push(Event::MouseMove {
                    x_diff,
                    y_diff,
                    x,
                    y,
                });
                self.context.mouse_position = Position { x, y };
            }
            event => {
                self.events.push(event);
            }
        }
    }

    pub fn should_redraw(&self) -> bool {
        true
    }

    pub fn on_window_create(&mut self, event_loop_proxy: EventLoopProxy<()>, size: Size) {
        self.window.size = size;
        self.event_loop_proxy = Some(event_loop_proxy);
        self.setup_file_watcher();
        self.load_widgets();
    }

    pub fn kill_processes(&mut self) {
        for (_, process) in self.processes.iter_mut() {
            process.kill();
        }
        self.processes.clear();
    }

    pub fn exit(&mut self) {
        self.save_widgets();
        self.kill_processes();
    }

    pub fn save_widgets(&mut self) {
        for widget in self.widget_store.iter_mut() {
            widget.save(&mut self.wasm_messenger);
        }
        let mut result = String::new();
        for widget in self.widget_store.iter() {
            if widget.ephemeral {
                continue;
            }
            let widget_serialized = &format!(
                "{};\n",
                ron::ser::to_string_pretty(widget, PrettyConfig::default()).unwrap()
            );
            // println!("widget_serialized: {}", widget_serialized);
            result.push_str(widget_serialized);
        }
        let mut file = File::create(&self.widget_config_path).unwrap();
        file.write_all(result.as_bytes()).unwrap();
    }

    pub fn mark_widget_dirty(&mut self, id: usize) {
        self.dirty_widgets.insert(id);
    }
}

// I need a way for processes to add widgets
// I don't want them to need to remove widgets by holding onto a reference or something
// It should work like react, always rendering the whole thing.
// At the same time, there are effects that processes might want to run and we need to distinguish between these.

// If a process gives me some output that are the widgets I should render
// how do I store them? If they clear that output, I should remove them
// But that implies that I have a way to identify them.

// I could instead each frame add the widgets of each process.
// Then I would have a mostly blank list of widgets

// You might also want to have widgets that outlive the process.
// Widgets that stay only when the processes output is saved might not be desirable.
// Or is it?
// The process itself can be gone, but the output can still be there.
// In fact, is there really a distinction? Is the widget not just the code that generates it?

// But it is possible that it would be better/faster
// to keep track of what widgets some output produced.
// I hold onto some ids. If the output is gone, I remove the widgets.
// If the output changes, I remove the widgets and add the new ones.
// I could then maybe do some sort of key based diffing algorithm.

// I also want the ability to start from data and make visualizations
// That data might be persisted or in  buffer
// Maybe I should start with some json and making visualizations?
// This is the biggest problem, the space is so big, I don't want to rabbit hole on something.

// Another thing I'm struggling with is what should be built in
// vs what should be extension based
// Maybe I should default to built-in and then move to extension afterwards?
// Probably will have more to show for it.

// I think for events, I should have a per frame list of events.
// I can do this while keeping the vec of events flat just by keeping an index
// Then update gets a list of events that happened on that frame
// and can respond to them.

// Example demos
// Build my own powerpoint
// Bind properties to each other
// Build a platformer, that uses widgets as the level editor
// YJS as an external addition
// Browser based stuff

// Ideas:
//
// Make a process widget. It is a file, not text.
// We can get via drag and drop.
// On click runs it and makes a TextPane of the output.
// We can track when that changes, parse the RON it produces
// and add widgets
// We could also parse events this way to inject events into the system
// I like RON in general. But should consider some other format. Sad about toml
// It should almost certainly parse json
// I need to have an event system for buffers changing. But also a per frame
// event system for reading from stdout
//
// Make my own react renderer that talks on a socket and produces widgets
//
// I need a way to track on click handlers
// But the question is what do they do?
// I'm guessing they push an event into the queue?
// But they might need to call a program
// I guess they can do that via an event in the queue
//
// How do I deal with top level UI? Right now I have a scene
// selector. Can I make that work generically?

// The idea of a compound widget does nothing right now
// Since I don't serialize it is literally meaningless
