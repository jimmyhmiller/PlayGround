use std::{
    collections::{HashSet, HashMap},
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::mpsc::{Receiver, Sender},
    thread,
    time::Duration, process::ChildStdout,
};

use crate::{
    event::Event,
    fps_counter::FpsCounter,
    keyboard::Modifiers,
    wasm_messenger::WasmMessenger,
    widget::{Color, Position, Size, Widget, WidgetData, WidgetId, WidgetStore},
};

use nonblock::NonBlockingReader;
use notify::{FsEventWatcher, RecursiveMode};

use notify_debouncer_mini::{new_debouncer, Debouncer};
use ron::ser::PrettyConfig;
use serde::{Serialize, Deserialize};
use skia_safe::{
    perlin_noise_shader,
    runtime_effect::ChildPtr,
    Data, Font, FontStyle, ISize, RuntimeEffect, Shader, Typeface,
};

pub struct Context {
    pub mouse_position: Position,
    pub left_mouse_down: bool,
    pub right_mouse_down: bool,
    pub cancel_click: bool,
    pub modifiers: Modifiers,
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub enum Value {
    USize(usize),
    F32(f32),
}

impl Value {
    #[allow(unused)]
    pub fn as_f32(&self) -> f32 {
        match self {
            Value::USize(v) => *v as f32,
            Value::F32(v) => *v,
        }
    }
}

pub enum PerFrame {
    ProcessOutput {
        process_id: usize,
    }
}

pub struct Process {
    pub process_id: usize,
    pub stdout: NonBlockingReader<ChildStdout>,
    pub stdin: std::process::ChildStdin,
    pub stderr: std::process::ChildStderr,
    pub output: String,
    pub widget_id: usize,
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
        self.external_sender = Some(sender_clone.clone());
        self.wasm_messenger.set_external_sender(sender_clone);
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
    }

    pub fn update(&mut self) {

        for widget in self.widget_store.iter_mut() {
            match &widget.data {
                WidgetData::Wasm { wasm: _, wasm_id } => {
                    self.wasm_messenger.send_update_position(*wasm_id, &widget.position);
                }
                _ => {}
            }
        }
        // Todo: Need to test that I am not missing any
        // events with my start and end

        // TODO: Put in better place
        for widget in self.widget_store.iter_mut() {
            match &widget.data {
                WidgetData::Wasm { wasm: _, wasm_id } => {
                    self.wasm_messenger.send_draw(*wasm_id, "draw");
                }
                _ => {}
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
                        if !buf.is_empty() {
                            process.output.push_str(&buf);
                        }
                    } else {
                        to_delete.insert(*process_id);
                    }
                }
            }
        }

        self.per_frame_actions
            .retain(|action| match action {
                PerFrame::ProcessOutput { process_id: id } => {
                    !to_delete.contains(id)
                },
            });

        let mut to_delete = vec![];

        for process in self.processes.values() {
            let widget_id = process.widget_id;
            let widget = self.widget_store.get_mut(widget_id).unwrap();
            let output = &process.output;
            match &mut widget.data {
                WidgetData::TextPane { text_pane } => {
                    text_pane.set_text(output);
                }
                // Shouldn't this be a process?
                WidgetData::Process { process: _ } => todo!(),
                WidgetData::Deleted => {
                    to_delete.push(process.process_id);
                }
                _ => unreachable!("Shouldn't be here")
            }
        }

        for process_id in to_delete {
            self.processes.remove(&process_id);
        }
        

        self.wasm_messenger.tick(&mut self.values);

        if let Some(receiver) = &self.external_receiver {
            for event in receiver.try_iter() {
                {
                    self.events.push_current_frame(event);
                }
            }
        }

        let events = self.events.events_for_frame().to_vec();

        if !events.is_empty() {
            self.should_redraw = true;
        }

        self.handle_events(events);
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
            wasm_messenger: WasmMessenger::new(None),
            widget_config_path: widget_config_path.to_string(),
            values: HashMap::new(),
            per_frame_actions: Vec::new(),
            processes: HashMap::new(),
        }
    }

    pub fn draw(&mut self, canvas: &mut Canvas) {
        self.fps_counter.tick();
        use skia_safe::Size;

        let background = Color::parse_hex("#39463e");
        // let darker = Color::parse_hex("#0d1d20");
        // let lighter = Color::parse_hex("#425050");
        let mut color = Color::parse_hex("#ffffff").to_color4f();
        color.a = 0.0;
        canvas.clear(background.to_color4f());

        let canvas_size = Size::from(canvas.base_layer_size());

        // let mut paint = Paint::new(darker.to_color4f(), None);
        // paint.set_anti_alias(true);
        // paint.set_style(PaintStyle::Fill);

        // let grain_shader = make_grain_gradient_shader(
        //     darker,
        //     lighter,
        //     0.95,
        //     crate::widget::Size {
        //         width: canvas_size.width,
        //         height: canvas_size.height,
        //     },
        // );

        // paint.set_shader(grain_shader);

        // paint.set_dither(true);


        // canvas.draw_rect(
        //     Rect::from_xywh(0.0, 0.0, canvas_size.width, canvas_size.height),
        //     &paint,
        // );

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

        canvas.save();
        canvas.translate((canvas_size.width - 300.0, 60.0));
        let counts = self.wasm_messenger.number_of_outstanding_messages();
        for line in counts.lines() {
            canvas.draw_str(line, Point::new(0.0, 0.0), &font, white);
            canvas.translate((0.0, 30.0));
        }
        canvas.restore();

        let mut to_draw = vec![];
        for widget in self.widget_store.iter_mut() {
            let before_count = canvas.save();
            to_draw.extend(widget.draw(canvas, &mut self.wasm_messenger, widget.size, &self.values));
            canvas.restore_to_count(before_count);
            canvas.restore();
        }

        // TODO: This is bad, I need to traverse the tree. Being lazy
        // Also not even sure about compound right now
        for widget_id in to_draw {
            if let Some(widget) = self.widget_store.get_mut(widget_id) {
                let before_count = canvas.save();
                widget.draw(canvas, &mut self.wasm_messenger, widget.size, &self.values);
                canvas.restore_to_count(before_count);
                canvas.restore();
            }
        }
    }

    pub fn add_event(&mut self, event: &winit::event::Event<'_, ()>) {
        if let Some(event) = Event::from_winit_event(event, self.context.modifiers) {
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
                            WidgetData::Wasm { wasm: _, wasm_id } => {
                                self.wasm_messenger.send_on_key(wasm_id, input);
                            }
                            _ => {}
                        }
                    }
                }
            }
            Event::ModifiersChanged(_) => {
                self.events.push(event);
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
            Event::StartProcess(_, _) => {
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

        let mut to_delete = vec![];

        let mut mouse_over = vec![];
        // I would need some sort of hierarchy here
        for widget in self.widget_store.iter_mut() {
            if widget.mouse_over(&self.context.mouse_position) {
                mouse_over.push(widget.id);

                let modifiers = self.context.modifiers;
                if modifiers.cmd && modifiers.ctrl && modifiers.option {
                    to_delete.push(widget.id);
                    continue;
                }

                let events =
                    widget.on_click(&self.context.mouse_position, &mut self.wasm_messenger);
                for event in events.iter() {
                    self.events.push(event.clone());
                }
            }
        }
        for id in mouse_over {
            self.respond_to_event(Event::WidgetMouseUp { widget_id: id });
        }
        for widget_id in to_delete {
            self.widget_store.delete_widget(widget_id);
        }

    }

    pub fn should_redraw(&self) -> bool {
        // self.should_redraw
        true
    }

    pub fn on_window_create(&mut self, event_loop_proxy: EventLoopProxy<()>) {
        self.event_loop_proxy = Some(event_loop_proxy);
        self.setup_file_watcher();
        self.load_widgets();
    }

    pub fn exit(&mut self) {
        self.save_widgets();
    }

    pub fn save_widgets(&mut self) {
        for widget in self.widget_store.iter_mut() {
            widget.save(&mut self.wasm_messenger);
        }
        let mut result = String::new();
        for widget in self.widget_store.iter() {
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
}

#[allow(unused)]
pub fn make_grain_gradient_shader(
    darker: Color,
    lighter: Color,
    alpha: f32,
    size: Size,
) -> Shader {
    let noise_shader = perlin_noise_shader::turbulence(
        (0.25, 0.25),
        1,
        0.0,
        ISize {
            width: size.width as i32 * 2,
            height: size.height as i32 * 2,
        },
    )
    .unwrap();

    let new_gradient_shader = RuntimeEffect::make_for_shader(
        "
        uniform vec2 iResolution;
        vec3 colorA = vec3(0.149,0.141,0.912);
        vec3 colorB = vec3(1.000,0.833,0.224);
        float alpha = 1.0;
        
        half4 main(vec2 fragCoord) {
          vec2 st = fragCoord / iResolution.xy;
          vec3 color = vec3(0.0);
        
          vec3 pct = vec3(st.x);
        
          // pct.r = smoothstep(0.0,1.0, st.x);
          pct.g = sin(st.x*3.14 + st.y*2.616) * -0.488;
          // pct.b = pow(st.x,0.5);
        
          color = mix(colorA, colorB, pct);
          return vec4(color.r * alpha, color.g * alpha, color.b * alpha, alpha);
        }
    ",
        None,
    )
    .unwrap();

    let effect = RuntimeEffect::make_for_shader(
        "
                uniform shader noiseShader;
                uniform shader gradientShader;
                uniform vec4 colorLight;
                uniform vec4 colorDark;
                uniform float alpha;
                
                half4 main(vec2 fragcoord) { 
                    vec4 noiseColor = noiseShader.eval(fragcoord);
                    vec4 gradientColor = gradientShader.eval(fragcoord);
                    // float noiseLuma = dot(noiseColor.rgb, vec3(0.299, 0.587, 0.114));
                    gradientColor.r += noiseColor.r * 0.4;
                    gradientColor.g += noiseColor.g * 0.4;
                    gradientColor.b += noiseColor.b * 0.4;
                    // vec4 white = vec4(1.0,1.0,1.0,1.0);
                    // vec4 lighter = mix(gradientColor, white, 0.3);
                    // vec4 black = vec4(0.0, 0.0, 0.0, 1);
                    // vec4 darker = mix(gradientColor, black, 0.7);
                    // float noiseLuma = dot(noiseColor.rgb, vec3(0.299, 0.587, 0.114));
                    // vec4 duotone = mix(gradientColor, darker, noiseLuma);
                    vec4 duotone = gradientColor;
                    return vec4(duotone.r * alpha, duotone.g * alpha, duotone.b * alpha, alpha);
                }

        ",
        None,
    )
    .unwrap();
    let data = [size.width, size.height];

    let len = data.len();
    let ptr = data.as_ptr() as *const u8;
    let data: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len * 4) };
    let new_gradient_shader = new_gradient_shader
        .make_shader(Data::new_copy(data), &[], None, false)
        .unwrap();
    let darker4 = darker.to_color4f();
    let lighter4 = lighter.to_color4f();
    let data = [
        darker4.r, darker4.g, darker4.b, darker4.a, lighter4.r, lighter4.g, lighter4.b, lighter4.a,
        alpha,
    ];

    let len = data.len();
    let ptr = data.as_ptr() as *const u8;
    let data: &[u8] = unsafe { std::slice::from_raw_parts(ptr, len * 4) };

    let uniforms = Data::new_copy(data);
    let grain_shader = effect
        .make_shader(
            uniforms,
            &[
                ChildPtr::Shader(noise_shader),
                ChildPtr::Shader(new_gradient_shader),
            ],
            None,
            false,
        )
        .unwrap();
    grain_shader
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
