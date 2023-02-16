use std::{
    collections::HashSet,
    fs::{File},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::mpsc::Receiver,
    thread, time::Duration,
};

use crate::{
    event::Event,
    fps_counter::FpsCounter,
    widget::{Color, Position, Size, TextPane, Wasm, Widget, WidgetData, WidgetId, WidgetStore},
};

use notify::{Config, RecommendedWatcher, RecursiveMode, Watcher, FsEventWatcher};

use notify_debouncer_mini::{new_debouncer, Debouncer};
use ron::ser::PrettyConfig;
use skia_safe::{Font, FontStyle, Typeface};

// Need a global store of widgets
// Then anything else refers to widgets by their id
// The nice part about that is that widgets can appear in multiple places
// It might make some other parts more awkward.
// I can also then deal with events to a widget as data with the id
// So things can listen for different interactions.
// If there are a large number of widgets, this could be a bottle neck
// So I could keep the hierachy. Like I know if the widget is not
// on the current scene then I am not going to click on it.

struct Context {
    mouse_position: Position,
    left_mouse_down: bool,
    right_mouse_down: bool,
    cancel_click: bool,
}

pub struct Editor {
    events: Events,
    pub fps_counter: FpsCounter,
    context: Context,
    widget_store: WidgetStore,
    should_redraw: bool,
    selected_widgets: HashSet<WidgetId>,
    active_widget: Option<WidgetId>,
    external_receiver: Option<Receiver<Event>>,
    debounce_watcher: Option<Debouncer<FsEventWatcher>>,
    event_loop_proxy: Option<EventLoopProxy<()>>,
    // points: Vec<[f64; 2]>,
}

struct Events {
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

    fn push(&mut self, event: Event) {
        self.events.push(event);
    }

    fn push_current_frame(&mut self, event: Event) {
        self.events.push(event);
        self.frame_end_index = self.events.len();
    }

    fn events_for_frame(&self) -> &[Event] {
        &self.events[self.frame_start_index..self.frame_end_index]
    }

    fn next_frame(&mut self) {
        self.frame_start_index = self.frame_end_index;
    }

    fn end_frame(&mut self) {
        self.frame_end_index = self.events.len();
    }
}

#[cfg(all(target_os = "macos"))]
use skia_safe::{Canvas, Color4f, Paint, Point};
use winit::event_loop::EventLoopProxy;

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
    }

    // TODO: Figure out a better setup for this
    // part of the problem is that scenes are not first class widgets
    // I need to figure out that hierarchy.
    // It will be useful for something like slideshow software
    pub fn setup(&mut self) {}

    fn setup_file_watcher(&mut self, widget_config_path: &str) {
        let widget_config_path = widget_config_path.to_string();
        let (watch_raw_send, watch_raw_receive) = std::sync::mpsc::channel();
        let (sender, receiver) = std::sync::mpsc::channel::<Event>();
        let mut debouncer : Debouncer<FsEventWatcher> = new_debouncer(Duration::from_millis(250), None, watch_raw_send).unwrap();


        let mut watcher = debouncer.watcher();

        // Probably need to debounce this
        watcher
            .watch(Path::new(&widget_config_path), RecursiveMode::NonRecursive)
            .unwrap();

        let event_loop_proxy = self.event_loop_proxy.as_ref().unwrap().clone();
        thread::spawn(move || {
            for res in watch_raw_receive {
                match res {
                    Ok(event) => {
                        event.iter().for_each(|event| {
                            let path = &event.path;
                            if path.to_str().unwrap() == widget_config_path {
                                sender.send(Event::ReloadWidgets).unwrap();
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
        self.debounce_watcher = Some(debouncer);
    }

    fn load_widgets(&mut self, path: &str) {
        let mut file = File::open(path).unwrap();
        let mut contents = String::new();
        file.read_to_string(&mut contents).unwrap();
        let widgets: Vec<Widget> = contents
            .split(";")
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(|s| {
                if let Ok(mut widget) = ron::de::from_str::<Widget>(s) {
                    widget.init();
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
    }

    pub fn update(&mut self) {
        // Todo: Need to test that I am not missing any
        // events with my start and end

        if let Some(receiver) = &self.external_receiver {
            for event in receiver.try_iter() {
                match event {
                    _ => {
                        self.events.push_current_frame(event);
                    }
                }
            }
        }

        let events = self.events.events_for_frame().to_vec();

        if !events.is_empty() {
            self.should_redraw = true;
        }

        for event in events {
            match event {
                Event::DroppedFile { path, x, y } => {
                    if path.extension().unwrap() == "wasm" {
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
                                WidgetData::Wasm { wasm } => {
                                    wasm.on_scroll(x, y);
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
                    self.load_widgets("widgets.ron");
                }
                Event::SaveWidgets => {
                    self.save_widgets("widgets.ron");
                }
                Event::ReloadWasm(path) => {
                    for widget in self.widget_store.iter_mut() {
                        if let WidgetData::Wasm { wasm } = &mut widget.data {
                            if path == wasm.path {
                                wasm.reload();
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    pub fn new() -> Self {
        Self {
            events: Events::new(),
            fps_counter: FpsCounter::new(),
            context: Context {
                mouse_position: Position { x: 0.0, y: 0.0 },
                left_mouse_down: false,
                right_mouse_down: false,
                cancel_click: false,
            },
            widget_store: WidgetStore::new(),
            should_redraw: true,
            selected_widgets: HashSet::new(),
            external_receiver: None,
            debounce_watcher: None,
            event_loop_proxy: None,
            active_widget: None,
        }
    }

    pub fn draw(&mut self, canvas: &mut Canvas) {
        self.fps_counter.tick();
        use skia_safe::Size;

        let gray = Color::parse_hex("#333333");
        canvas.clear(gray.to_color4f());

        // let points = &self.points;
        // let stroke_points = get_stroke_points(points.clone(), StrokeOptions::new());
        // draw_strokes(&stroke_points, canvas);

        // let outline_points = get_stroke_outline_points(&stroke_points.clone(), StrokeOptions::new());
        // draw_points(&outline_points, canvas);


        let canvas_size = Size::from(canvas.base_layer_size());

        let font = Font::new(
            Typeface::new("Ubuntu Mono", FontStyle::bold()).unwrap(),
            32.0,
        );
        let white = &Paint::new(Color4f::new(1.0, 1.0, 1.0, 1.0), None);

        canvas.draw_str(
            self.fps_counter.fps.to_string(),
            Point::new(canvas_size.width - 60.0, 30.0),
            &font,
            white,
        );

        let mut to_draw = vec![];
        for widget in self.widget_store.iter_mut() {
            to_draw.extend(widget.draw(canvas));
        }

        // TODO: This is bad, I need to traverse the tree. Being lazy
        // Also not even sure about compound right now
        for widget_id in to_draw {
            if let Some(widget) = self.widget_store.get_mut(widget_id) {
                widget.draw(canvas);
            }
        }
    }

    pub fn add_event(&mut self, event: &winit::event::Event<'_, ()>) {
        if let Some(event) = Event::from_winit_event(event) {
            self.respond_to_event(event);
        }
    }

    fn respond_to_event(&mut self, mut event: Event) {
        event.patch_mouse_event(&self.context.mouse_position);
        match event {
            Event::WidgetMouseDown { widget_id: _ } => {
                self.events.push(event);
            }
            Event::WidgetMouseUp { widget_id: _ } => {
                self.events.push(event);
            }
            Event::Noop => {}
            Event::KeyEvent { input } => {
                self.events.push(event);
                if let Some(widget_id) = self.active_widget {
                    if let Some(widget) = self.widget_store.get_mut(widget_id) {
                        match widget.data {
                            WidgetData::Wasm { ref mut wasm } => {
                                wasm.on_key(input);
                            }
                            _ => {}
                        }
                    }
                }
            }
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
                // self.points.push([x as f64, y as f64])
            }
            Event::LeftMouseDown { .. } => {
                self.events.push(event);
                self.context.left_mouse_down = true;
                self.context.cancel_click = false;
                self.add_mouse_down();
            }
            Event::LeftMouseUp { .. } => {
                self.events.push(event);
                // Probably not the right place.
                // Maybe need events on last cycle?
                self.context.left_mouse_down = false;
                self.add_mouse_up();
                // self.points = vec![];
            }
            Event::RightMouseDown { .. } => {
                self.events.push(event);
                self.context.right_mouse_down = true;
            }
            Event::RightMouseUp { .. } => {
                self.events.push(event);
                self.context.right_mouse_down = false;
            }
            Event::Scroll { x: _, y: _ } => {
                self.events.push(event);
            }
            Event::HoveredFile {
                path: _,
                x: _,
                y: _,
            } => {
                self.events.push(event);
            }
            Event::DroppedFile {
                path: _,
                x: _,
                y: _,
            } => {
                self.events.push(event);
            }
            Event::HoveredFileCancelled => {
                self.events.push(event);
            }
            Event::MoveWidgetRelative { .. } => {}
            Event::ReloadWidgets => {
                self.events.push(event);
            }
            Event::SaveWidgets => {
                self.events.push(event);
            }
            Event::ReloadWasm(_) => {
                self.events.push(event);
            }
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
        for widget in self.widget_store.iter() {
            if widget.mouse_over(&self.context.mouse_position) {
                found_a_widget = true;
                mouse_over.push(widget.id);
                self.selected_widgets.insert(widget.id);
                // TODO: This is ugly, just setting the active widget
                // over and over again then we will get the last one
                // which would probably draw on top anyways.
                // Should do better
                self.active_widget = Some(widget.id);
            }
            if !found_a_widget {
                self.active_widget = None;
            }
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
        if self.context.cancel_click {
            self.context.cancel_click = false;
            return;
        }

        let mut mouse_over = vec![];
        // I would need some sort of hierarchy here
        for widget in self.widget_store.iter_mut() {
            if widget.mouse_over(&self.context.mouse_position) {
                mouse_over.push(widget.id);
                let events = widget.on_click(&self.context.mouse_position);
                for event in events.iter() {
                    self.events.push(event.clone());
                }
            }
        }
        for id in mouse_over {
            self.respond_to_event(Event::WidgetMouseUp { widget_id: id });
        }
    }

    pub fn should_redraw(&self) -> bool {
        self.should_redraw
        // true
    }

    pub fn on_window_create(&mut self, event_loop_proxy: EventLoopProxy<()>) {
        self.event_loop_proxy = Some(event_loop_proxy);
        let mut widget_config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        widget_config_path.push("widgets.ron");
        let widget_config_path = widget_config_path.to_str().unwrap();
        self.setup_file_watcher(widget_config_path);
        self.load_widgets(widget_config_path);
    }

    pub fn exit(&mut self) {
        for widget in self.widget_store.iter_mut() {
            widget.save();
        }
        self.save_widgets("widgets.ron");
    }

    fn save_widgets(&mut self, path: &str) {
        let mut file = File::create(path).unwrap();
        let widgets = self
            .widget_store
            .iter_mut()
            .map(|widget| {
                widget.save();
                widget
            })
            .collect::<Vec<_>>();
        let mut result = String::new();
        for widget in widgets.iter() {
            result.push_str(&format!(
                "{};\n",
                ron::ser::to_string_pretty(widget, PrettyConfig::default()).unwrap()
            ));
        }

        file.write_all(result.as_bytes()).unwrap();
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
