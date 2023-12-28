use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ChildStdout,
    sync::mpsc::{Receiver, Sender},
    thread,
    time::{Duration, Instant},
};

use crate::{
    color::Color,
    event::Event,
    fps_counter::FpsCounter,
    keyboard::Modifiers,
    wasm_messenger::WasmMessenger,
    widget::{Widget, WidgetId, WidgetStore},
    widget2::{Deleted, TextPane, Widget as Widget2},
};

use framework::{Position, Size, Value};
use itertools::Itertools;
use nonblock::NonBlockingReader;
use notify::{FsEventWatcher, RecursiveMode};

use notify_debouncer_mini::{new_debouncer, Debouncer};

use serde::{Deserialize, Serialize};
use skia_safe::Canvas;
use winit::{event_loop::EventLoopProxy, window::CursorIcon};

pub struct Context {
    pub raw_mouse_position: Position,
    pub mouse_position: Position,
    pub left_mouse_down: bool,
    pub right_mouse_down: bool,
    pub cancel_click: bool,
    pub modifiers: Modifiers,
    pub moved: HashSet<usize>,
}

pub enum PerFrame {
    ProcessOutput { process_id: usize },
}

#[derive(Serialize, Deserialize)]
struct SavedOutput {
    widgets: serde_json::Value,
    values: HashMap<String, Value>,
}

#[derive(Serialize, Deserialize)]
struct SavedInput {
    widgets: Vec<Widget>,
    values: HashMap<String, Value>,
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
    pub first_frame: bool,
    pub canvas_scroll_offset: Position,
    pub canvas_scale: f32,
    pub show_debug: bool,
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

    pub fn push_current_frame(&mut self, event: Event) {
        self.events.push(event);
        self.frame_end_index = self.events.len();
    }

    pub fn events_for_frame(&self) -> &[Event] {
        &self.events[self.frame_start_index..self.frame_end_index]
    }

    pub fn newly_added_events(&self) -> &[Event] {
        &self.events[self.frame_end_index..]
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

        let input: SavedInput = serde_json::from_str(&contents).unwrap();
        let widgets = input.widgets;
        self.values = input.values;

        let widgets: Vec<Widget> = widgets
            .into_iter()
            .map(|mut widget| {
                widget.init(
                    &mut self.wasm_messenger,
                    self.values.clone(),
                    self.external_sender.as_ref().unwrap().clone(),
                );
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
            })
            .collect();

        for mut widget in widgets {
            let id = self.widget_store.next_id();
            widget.set_id(id);
            self.widget_store.add_widget(widget);
        }
        for widget in self.widget_store.iter_mut() {
            if let Some(widget) = widget.as_wasm_widget_mut() {
                widget.external_sender = Some(self.external_sender.as_ref().unwrap().clone());
            }
            // if widget.position().x >= self.window.size.width
            //     || widget.position().y >= self.window.size.height
            // {
            //     println!(
            //         "Widget out of bounds, moving to edge of screen {}",
            //         widget.id()
            //     );
            //     let x = widget
            //         .position()
            //         .x
            //         .min(self.window.size.width - widget.size().width * widget.scale());
            //     let y = widget
            //         .position()
            //         .y
            //         .min(self.window.size.height - widget.size().height * widget.scale());

            //     widget.data.on_move(x, y).unwrap();
            // }
        }
    }

    pub fn update(&mut self) -> bool {
        // TODO: Put in better place

        // Todo: Need to test that I am not missing any
        // events with my start and end

        let time = Instant::now();
        self.wasm_messenger.tick();
        self.fps_counter.add_time("tick", time.elapsed());

        let events = self.events.events_for_frame().to_vec();

        let events_empty = events.is_empty();

        if !events_empty {
            self.should_redraw = true;
        }

        let time = Instant::now();
        self.handle_events(events);
        self.wasm_messenger.tick();
        if let Some(receiver) = &self.external_receiver {
            for event in receiver.try_iter() {
                self.events.push(event);
            }
        }
        let newly_added_events = self.events.newly_added_events().to_vec();
        self.handle_events(newly_added_events);
        self.events.end_frame();
        self.next_frame();
        self.fps_counter.add_time("events", time.elapsed());

        let time = Instant::now();
        for widget in self.widget_store.iter_mut() {
            if widget.dirty() {
                widget.update().unwrap();
            } else if let Some(widget) = widget.as_wasm_widget_mut() {
                if widget.draw_commands.is_empty() {
                    widget.update().unwrap();
                }
            }
        }
        self.fps_counter.add_time("update_widgets", time.elapsed());

        let pending_count: usize = self
            .widget_store
            .iter()
            .filter_map(|x| x.as_wasm_widget())
            .map(|x| x.number_of_pending_requests())
            .sum();

        if pending_count > 0 {
            self.should_redraw = true;
        }

        if self.widget_store.iter().any(|x| x.dirty()) {
            self.should_redraw = true;
        }

        !events_empty || pending_count > 0
    }

    pub fn process_per_frame_actions(&mut self) {
        if let Some(receiver) = &self.external_receiver {
            for event in receiver.try_iter() {
                self.events.push_current_frame(event);
            }
        }

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
                        // TODO: need to send stderr separately
                        // let stderr = &mut process.stderr;
                        // let max_attempts = 100;
                        // let mut i = 0;
                        // while !stderr.is_eof() {
                        //     if i > max_attempts {
                        //         break;
                        //     }
                        //     let length = stderr.read_available_to_string(&mut buf).unwrap();
                        //     if length == 0 {
                        //         break;
                        //     }
                        //     i += 1;
                        // }
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
                // TODO: crash here on reload
                let widget = self.widget_store.get_mut(widget_id).unwrap();
                widget.mark_dirty("process_output");
                let output = &process.output;

                if let Some(widget) = widget.data.as_any_mut().downcast_mut::<TextPane>() {
                    widget.set_text(output);
                }
                if widget.data.as_any().downcast_ref::<Deleted>().is_some() {
                    to_delete.insert(process.process_id);
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
        widget_config_path.push("widgets.json");
        let widget_config_path = widget_config_path.to_str().unwrap();

        Self {
            events: Events::new(),
            fps_counter: FpsCounter::new(),
            context: Context {
                raw_mouse_position: Position { x: 0.0, y: 0.0 },
                mouse_position: Position { x: 0.0, y: 0.0 },
                left_mouse_down: false,
                right_mouse_down: false,
                cancel_click: false,
                modifiers: Modifiers::default(),
                moved: HashSet::new(),
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
            first_frame: true,
            canvas_scroll_offset: Position { x: 0.0, y: 0.0 },
            canvas_scale: 1.0,
            show_debug: false,
        }
    }

    // IDEA: Draw on back with negative coordinates
    pub fn draw(&mut self, canvas: &Canvas) {
        self.fps_counter.tick();
        use skia_safe::{Color4f, Font, FontStyle, Paint, Point, Typeface};

        let background = Color::parse_hex("#39463e");

        canvas.clear(background.as_color4f());

        let font = Font::new(
            Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(),
            32.0,
        );
        let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);

        let canvas_size = self.window.size;

        if self.show_debug {
            canvas.draw_str(
                self.fps_counter.fps.to_string(),
                Point::new(canvas_size.width - 60.0, 30.0),
                &font,
                white,
            );
            canvas.save();
            canvas.translate((canvas_size.width - 600.0, 60.0));
            for (name, stats) in self.fps_counter.times.iter() {
                canvas.draw_str(
                    format!(
                        "{}: {} {} {}",
                        name,
                        stats.average.as_millis(),
                        stats.min.as_millis(),
                        stats.max.as_millis()
                    ),
                    Point::new(0.0, 0.0),
                    &font,
                    white,
                );
                canvas.translate((0.0, 30.0));
            }
            canvas.restore();

            let mut combined_counts: HashMap<String, usize> = HashMap::new();
            for widget in self.widget_store.iter() {
                if let Some(widget) = widget.as_wasm_widget() {
                    let counts = widget.pending_message_counts();
                    for (key, value) in counts.iter() {
                        let count = combined_counts.entry(key.to_string()).or_insert(0);
                        *count += value;
                    }
                }
            }

            canvas.save();
            canvas.translate((canvas_size.width - 1000.0, 60.0));
            let mut output = String::new();
            for (category, count) in combined_counts.iter().sorted() {
                output.push_str(&format!("{} : {}\n", category, count));
            }
            for line in output.lines() {
                canvas.draw_str(line, Point::new(0.0, 0.0), &font, white);
                canvas.translate((0.0, 30.0));
            }
            canvas.restore();
        }

        canvas.translate((self.canvas_scroll_offset.x, self.canvas_scroll_offset.y));
        canvas.scale((self.canvas_scale, self.canvas_scale));

        self.widget_store.fix_zindexes();

        let dirty_widgets: HashSet<usize> = self
            .widget_store
            .iter()
            .filter(|x| x.dirty())
            .map(|x| x.id())
            .collect();
        self.widget_store.draw(canvas, &dirty_widgets);

        for widget in self.widget_store.iter_mut() {
            widget.reset_dirty();
        }
    }

    pub fn add_event(&mut self, event: &winit::event::Event<()>) -> bool {
        if let Some(event) = Event::from_winit_event(event, self.context.modifiers) {
            self.respond_to_event(event);
            return true;
        }
        false
    }

    // TODO: Do I need this indirection?
    pub fn respond_to_event(&mut self, mut event: Event) {
        event.patch_mouse_event(
            &self.context.mouse_position,
            &self.canvas_scroll_offset,
            self.canvas_scale,
        );
        match event {
            Event::Noop => {}
            Event::MouseMove { x, y, .. } => {
                // I want to be able to respond to mouse move events
                // I just might not want to save them?
                // I'm not sure...
                self.context.raw_mouse_position = Position { x, y };
                let x = (x - self.canvas_scroll_offset.x) / self.canvas_scale;
                let y = (y - self.canvas_scroll_offset.y) / self.canvas_scale;
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
        self.should_redraw
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
        let saved: SavedOutput = SavedOutput {
            widgets: serde_json::to_value(
                self.widget_store
                    .iter()
                    .filter(|x| x.typetag_name() != "Ephemeral" && x.typetag_name() != "Deleted")
                    .collect_vec(),
            )
            .unwrap(),
            values: self.values.clone(),
        };
        let result = serde_json::ser::to_string_pretty(&saved).unwrap();
        let mut file = File::create("widgets.json").unwrap();
        file.write_all(result.as_bytes()).unwrap();
    }

    pub fn mark_widget_dirty(&mut self, id: usize, reason: &str) {
        if let Some(widget) = self.widget_store.get_mut(id) {
            widget.mark_dirty(reason);
        }
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
