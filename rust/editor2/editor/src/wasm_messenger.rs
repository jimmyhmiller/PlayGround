use std::{
    collections::HashMap,
    error::Error,
    path::Path,
    sync::{mpsc, Arc},
    thread,
    time::Duration, os::unix::process,
};

use bytesize::ByteSize;

use futures::{
    channel::{
        mpsc::{channel, Receiver, Sender},
        oneshot,
    },
    executor::{LocalPool, LocalSpawner},
    task::LocalSpawnExt,
    StreamExt,
};
use futures_timer::Delay;
use itertools::Itertools;

use skia_safe::{Canvas, Font, FontStyle, Typeface};
use wasmtime::{
    AsContextMut, Caller, Config, Engine, Instance, Linker, Memory, Module, Store, Val, WasmParams,
    WasmResults,
};
use wasmtime_wasi::{Dir, WasiCtxBuilder};

use crate::{
    editor::Value,
    event::Event,
    keyboard::KeyboardInput,
    widget::{Color, Position, Size},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct PointerLengthString {
    ptr: u32,
    len: u32,
}

pub type WasmId = u64;

#[derive(Debug, Clone)]
enum Payload {
    #[allow(unused)]
    NewInstance(String),
    OnClick(Position),
    Draw(String),
    SetState(String),
    OnScroll(f64, f64),
    OnKey(KeyboardInput),
    Reload,
    SaveState,
    UpdatePosition(Position),
    ProcessMessage(usize, String),
}

#[derive(Clone, Debug)]
struct Message {
    message_id: usize,
    wasm_id: WasmId,
    payload: Payload,
}

enum OutPayload {
    DrawCommands(Vec<Command>),
    Saved(SaveState),
    ErrorPayload(String),
    Complete,
    NeededValue(String, oneshot::Sender<String>),
}

struct OutMessage {
    message_id: usize,
    wasm_id: WasmId,
    payload: OutPayload,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SaveState {
    Unsaved,
    Empty,
    Saved(String),
}

pub struct WasmMessenger {
    local_pool: futures::executor::LocalPool,
    local_spawner: LocalSpawner,
    last_wasm_id: u64,
    wasm_draw_commands: HashMap<WasmId, Vec<Command>>,
    wasm_non_draw_commands: HashMap<WasmId, Vec<Command>>,
    wasm_states: HashMap<WasmId, SaveState>,
    last_message_id: usize,
    // Not a huge fan of this solution,
    // but couldn't find a better way to dedup draws
    // Ideally, you can draw in the middle of click commands
    // I have some ideas.
    outstanding_messages: HashMap<WasmId, HashMap<usize, Message>>,
    engine: Arc<Engine>,
    receivers: HashMap<WasmId, Receiver<OutMessage>>,
    senders: HashMap<WasmId, Sender<Message>>,
    external_sender: Option<mpsc::Sender<Event>>,
}

impl WasmMessenger {
    pub fn new(external_sender: Option<mpsc::Sender<Event>>) -> Self {
        let local_pool = LocalPool::new();
        let local_spawner = local_pool.spawner();

        let mut config = Config::new();
        config.dynamic_memory_guard_size(ByteSize::mb(500).as_u64());
        config.static_memory_guard_size(ByteSize::mb(500).as_u64());
        config.epoch_interruption(true);
        config.async_support(true);

        let engine = Arc::new(Engine::new(&config).unwrap());

        let engine_clone = engine.clone();
        thread::spawn(move || loop {
            thread::sleep(Duration::from_millis(4));
            engine_clone.increment_epoch();
        });

        Self {
            local_pool,
            local_spawner,
            last_wasm_id: 0,
            wasm_draw_commands: HashMap::new(),
            wasm_non_draw_commands: HashMap::new(),
            wasm_states: HashMap::new(),
            last_message_id: 0,
            outstanding_messages: HashMap::new(),
            engine,
            receivers: HashMap::new(),
            senders: HashMap::new(),
            external_sender,
        }
    }

    pub fn set_external_sender(&mut self, external_sender: mpsc::Sender<Event>) {
        self.external_sender = Some(external_sender);
    }

    pub fn number_of_outstanding_messages(&self) -> String {
        let mut stats: Vec<&str> = vec![];
        for messages_per in self.outstanding_messages.values() {
            for message in messages_per.values() {
                stats.push(match message.payload {
                    Payload::NewInstance(_) => "NewInstance",
                    Payload::OnClick(_) => "OnClick",
                    Payload::Draw(_) => "Draw",
                    Payload::SetState(_) => "SetState",
                    Payload::OnScroll(_, _) => "OnScroll",
                    Payload::OnKey(_) => "OnKey",
                    Payload::Reload => "Reload",
                    Payload::SaveState => "SaveState",
                    Payload::UpdatePosition(_) => "UpdatePosition",
                    Payload::ProcessMessage(_, _) => "ProcessMessage",
                });
            }
        }

        let mut output = String::new();
        let counts = stats.iter().counts();

        for (category, count) in counts.iter().sorted() {
            output.push_str(&format!("{} : {}\n", category, count));
        }

        output
    }

    fn next_message_id(&mut self) -> usize {
        self.last_message_id += 1;
        self.last_message_id
    }

    fn next_wasm_id(&mut self) -> WasmId {
        self.last_wasm_id += 1;
        self.last_wasm_id
    }

    pub fn new_instance(&mut self, wasm_path: &str) -> WasmId {
        let id = self.next_wasm_id();

        let (sender, receiver) = channel::<Message>(100000);
        let (out_sender, out_receiver) = channel::<OutMessage>(100000);

        self.receivers.insert(id, out_receiver);
        self.senders.insert(id, sender);

        async fn spawn_instance(
            engine: Arc<Engine>,
            wasm_id: WasmId,
            wasm_path: String,
            receiver: Receiver<Message>,
            sender: Sender<OutMessage>,
        ) {
            let mut instance = WasmManager::new(
                engine.clone(),
                wasm_id,
                wasm_path.to_string(),
                receiver,
                sender,
            )
            .await;
            instance.init().await;
        }

        self.local_spawner
            .spawn_local(spawn_instance(
                self.engine.clone(),
                id,
                wasm_path.to_string(),
                receiver,
                out_sender,
            ))
            .unwrap();

        id
    }

    pub fn draw_widget(
        &mut self,
        wasm_id: WasmId,
        canvas: &mut Canvas,
        bounds: Size,
    ) -> Option<Size> {
        if let Some(commands) = self.wasm_draw_commands.get(&wasm_id) {
            let mut max_width = 0.0;
            let mut max_height = 0.0;
            let mut current_height_stack = vec![];

            let mut paint = skia_safe::Paint::default();
            let mut non_draw_commands = vec![];
            for command in commands.iter() {
                match command {
                    Command::SetColor(r, g, b, a) => {
                        let color = Color::new(*r, *g, *b, *a);
                        paint.set_color(color.to_color4f().to_color());
                    }
                    Command::DrawRect(x, y, width, height) => {
                        canvas
                            .draw_rect(skia_safe::Rect::from_xywh(*x, *y, *width, *height), &paint);
                        // This is not quite right because of translate and stuff.
                        // if *x + *width > max_width {
                        //     max_width = *x + *width;
                        // }
                        // if *y + *height > max_height {
                        //     max_height = *y + *height;
                        // }
                    }
                    Command::DrawString(str, x, y) => {
                        let mut paint = paint.clone();
                        paint.set_shader(None);
                        if max_height > bounds.height {
                            continue;
                        }
                        let font = Font::new(
                            Typeface::new("Ubuntu Mono", FontStyle::normal()).unwrap(),
                            32.0,
                        );

                        // No good way right now to find bounds. Need to think about this properly
                        canvas.draw_str(str, (*x, *y), &font, &paint);
                    }

                    Command::ClipRect(x, y, width, height) => {
                        canvas.clip_rect(
                            skia_safe::Rect::from_xywh(*x, *y, *width, *height),
                            None,
                            None,
                        );
                        // if *width > max_width {
                        //     max_width = *width;
                        // }
                        // if *height > max_height {
                        //     max_height = *height;
                        // }
                    }
                    Command::DrawRRect(x, y, width, height, radius) => {
                        let rrect = skia_safe::RRect::new_rect_xy(
                            skia_safe::Rect::from_xywh(*x, *y, *width, *height),
                            *radius,
                            *radius,
                        );
                        canvas.draw_rrect(rrect, &paint);
                    }
                    Command::Translate(x, y) => {
                        max_height += *y;
                        max_width += *x;
                        canvas.translate((*x, *y));
                    }
                    Command::Save => {
                        canvas.save();
                        current_height_stack.push(max_height);
                    }
                    Command::Restore => {
                        canvas.restore();
                        max_height = current_height_stack.pop().unwrap();
                    }
                    c => {
                        non_draw_commands.push(c.clone());
                    }
                }
            }
            self.wasm_non_draw_commands
                .insert(wasm_id, non_draw_commands);
            Some(Size {
                width: max_width,
                height: max_height,
            })
        } else {
            None
        }
    }

    pub fn process_non_draw_commands(&mut self, values: &mut HashMap<String, Value>) {
        for (wasm_id, commands) in self.wasm_non_draw_commands.iter() {
            for command in commands.iter() {
                match command {
                    Command::Restore => println!("Unhandled"),
                    Command::Save => println!("Unhandled"),
                    Command::StartProcess(process_id, process_command) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::StartProcess(
                                *process_id as usize,
                                // TODO: I probably actually want widget id?
                                *wasm_id as usize,
                                process_command.clone(),
                            ))
                            .unwrap();
                    }
                    Command::SendProcessMessage(process_id, message) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::SendProcessMessage(
                                *process_id as usize,
                                message.clone(),
                            ))
                            .unwrap();
                    }
                    Command::ReceiveLastProcessMessage(_) => println!("Unhandled"),
                    Command::ProvideF32(name, val) => {
                        values.insert(name.to_string(), Value::F32(*val));
                    }
                    _ => println!("Draw command ended up here"),
                }
            }
        }
        self.wasm_non_draw_commands.clear();
    }

    pub fn tick(&mut self, values: &mut HashMap<String, Value>) {
        self.process_non_draw_commands(values);
        self.local_pool
            .run_until(Delay::new(Duration::from_millis(4)));

        // I need to do this slightly differently because I need to draw in the context
        // of the widget.
        // But on tick I could get the pending drawings and then draw them
        // for each widget

        // TODO: need to time this out
        for out_receiver in self.receivers.values_mut() {
            while let Ok(Some(message)) = out_receiver.try_next() {
                if let Some(record) = self.outstanding_messages.get_mut(&message.wasm_id) {
                    record.remove(&message.message_id);
                }

                match message.payload {
                    OutPayload::DrawCommands(commands) => {
                        self.wasm_draw_commands.insert(message.wasm_id, commands);
                    }
                    OutPayload::Saved(saved) => {
                        self.wasm_states.insert(message.wasm_id, saved);
                    }
                    OutPayload::ErrorPayload(error_message) => {
                        println!("Error: {}", error_message);
                    }
                    OutPayload::NeededValue(name, sender) => {
                        // If I don't have the value, what should I do?
                        // Should I save this message and re-enqueue or signal failure?
                        if let Some(value) = values.get(&name) {
                            let serialized = serde_json::to_string(value).unwrap();
                            sender.send(serialized).unwrap();
                        } else {
                            println!("Can't find value {}", name);
                        }
                    }
                    OutPayload::Complete => {}
                }
            }
        }
    }

    fn send_message(&mut self, message: Message) {
        let records = self
            .outstanding_messages
            .entry(message.wasm_id)
            .or_insert(HashMap::new());

        let mut already_drawing = false;
        if matches!(message.payload, Payload::Draw(_)) {
            for record in records.values() {
                if matches!(record.payload, Payload::Draw(_)) {
                    already_drawing = true;
                    break;
                }
            }
        }
        if !already_drawing {
            records.insert(message.message_id, message.clone());

            if let Some(sender) = self.senders.get_mut(&message.wasm_id) {
                sender.start_send(message).unwrap();
            } else {
                println!("Can't find wasm instance for message {:?}", message);
            }
        }
    }

    pub fn send_on_click(&mut self, wasm_id: WasmId, position: &Position) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::OnClick(*position),
        });
    }

    pub fn send_update_position(&mut self, wasm_id: WasmId, position: &Position) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::UpdatePosition(*position),
        });
    }

    pub fn send_draw(&mut self, wasm_id: WasmId, fn_name: &str) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::Draw(fn_name.to_string()),
        });
    }

    pub fn send_set_state(&mut self, wasm_id: WasmId, state: &str) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::SetState(state.to_string()),
        });
    }

    pub fn send_on_scroll(&mut self, wasm_id: u64, x: f64, y: f64) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::OnScroll(x, y),
        });
    }

    pub fn send_on_key(&mut self, wasm_id: u64, input: KeyboardInput) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::OnKey(input),
        });
    }

    pub fn send_reload(&mut self, wasm_id: u64) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::Reload,
        });
    }

    pub fn send_process_message(&mut self, wasm_id: u64, process_id: usize, buf: &str) {
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::ProcessMessage(process_id, buf.to_string()),
        });
    }

    // Sometimes state is corrupt by going too long on the string. Not sure why.
    // Need to track down the issue
    pub fn save_state(&mut self, wasm_id: WasmId) -> SaveState {
        self.wasm_states.insert(wasm_id, SaveState::Unsaved);
        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id,
            payload: Payload::SaveState,
        });
        // TODO: Maybe not the best design, but I also want to ensure I save
        loop {
            // TODO: Fix this
            self.tick(&mut HashMap::new());
            if let Some(state) = self.wasm_states.get(&wasm_id) {
                match state {
                    SaveState::Saved(state) => {
                        if state.starts_with('\"') {
                            assert!(state.ends_with('\"'), "State is corrupt: {}", state);
                        }
                        break;
                    }
                    SaveState::Empty => {
                        break;
                    }
                    _ => {}
                }
            }
        }
        self.wasm_states
            .get(&wasm_id)
            .unwrap_or(&SaveState::Empty)
            .clone()
    }


}

// I think I need to:
// 1. Spawn a task per wasm instance
// 2. Have senders and receivers per instance

struct WasmManager {
    #[allow(unused)]
    id: WasmId,
    instance: WasmInstance,
    receiver: Receiver<Message>,
    #[allow(unused)]
    engine: Arc<Engine>,
    sender: Sender<OutMessage>,
}

impl WasmManager {
    pub async fn new(
        engine: Arc<Engine>,
        wasm_id: WasmId,
        wasm_path: String,
        receiver: Receiver<Message>,
        sender: Sender<OutMessage>,
    ) -> Self {
        let instance = WasmInstance::new(engine.clone(), &wasm_path, sender.clone())
            .await
            .unwrap();

        Self {
            id: wasm_id,
            instance,
            receiver,
            engine,
            sender,
        }
    }

    pub async fn init(&mut self) {
        loop {
            let message = self.receiver.select_next_some().await;
            let out_message = self.process_message(message).await;
            self.sender.start_send(out_message).unwrap();
        }
    }

    pub async fn process_message(&mut self, message: Message) -> OutMessage {
        let id = message.wasm_id;
        let default_return = OutMessage {
            wasm_id: message.wasm_id,
            message_id: message.message_id,
            payload: OutPayload::Complete,
        };

        match message.payload {
            Payload::NewInstance(_) => {
                panic!("Shouldn't get here")
            }
            Payload::OnClick(position) => {
                self.instance
                    .on_click(position.x, position.y)
                    .await
                    .unwrap();
                default_return
            }
            Payload::Draw(fn_name) => {
                let result = self.instance.draw(&fn_name).await;
                match result {
                    Ok(result) => OutMessage {
                        message_id: message.message_id,
                        wasm_id: id,
                        payload: OutPayload::DrawCommands(result),
                    },
                    Err(error) => {
                        println!("Error drawing {:?}", error);
                        default_return
                    }
                }
            }
            Payload::SetState(state) => {
                self.instance.set_state(state.as_bytes()).await.unwrap();
                default_return
            }
            Payload::OnScroll(x, y) => {
                self.instance.on_scroll(x, y).await.unwrap();
                default_return
            }
            Payload::ProcessMessage(process_id, message) => {
                self.instance.on_process_message(process_id as i32, message).await.unwrap();
                default_return
            }
            Payload::OnKey(input) => {
                let (key_code, state, modifiers) = input.to_u32_tuple();
                let result = self.instance.on_key(key_code, state, modifiers).await;
                match result {
                    Ok(_) => default_return,
                    Err(err) => OutMessage {
                        wasm_id: message.wasm_id,
                        message_id: message.message_id,
                        payload: OutPayload::ErrorPayload(err.to_string()),
                    },
                }
            }
            Payload::Reload => {
                match self.instance.reload().await {
                    Ok(_) => {}
                    Err(e) => {
                        println!("Error reloading {}", e);
                    }
                }
                default_return
            }
            Payload::SaveState => {
                let state = self.instance.get_state().await;
                match state {
                    Some(state) => {
                        if state.starts_with('\"') {
                            assert!(state.ends_with('\"'), "State is corrupt: {}", state);
                        }
                        OutMessage {
                            message_id: message.message_id,
                            wasm_id: id,
                            payload: OutPayload::Saved(SaveState::Saved(state)),
                        }
                    }
                    None => OutMessage {
                        message_id: message.message_id,
                        wasm_id: id,
                        payload: OutPayload::Saved(SaveState::Empty),
                    },
                }
            }
            Payload::UpdatePosition(position) => {
                self.instance.store.data_mut().position = position;
                default_return
            }
        }
    }
}

struct State {
    wasi: wasmtime_wasi::WasiCtx,
    commands: Vec<Command>,
    get_state_info: (u32, u32),
    // Probably not the best structure
    // but lets start here
    process_messages: HashMap<i32, String>,
    position: Position,
    sender: Sender<OutMessage>,
}

impl State {
    fn new(wasi: wasmtime_wasi::WasiCtx, sender: Sender<OutMessage>) -> Self {
        Self {
            wasi,
            commands: Vec::new(),
            process_messages: HashMap::new(),
            get_state_info: (0, 0),
            position: Position { x: 0.0, y: 0.0 },
            sender,
        }
    }
}

#[derive(Debug, Clone)]
enum Command {
    DrawRect(f32, f32, f32, f32),
    DrawString(String, f32, f32),
    ClipRect(f32, f32, f32, f32),
    DrawRRect(f32, f32, f32, f32, f32),
    Translate(f32, f32),
    SetColor(f32, f32, f32, f32),
    Restore,
    Save,
    StartProcess(u32, String),
    SendProcessMessage(i32, String),
    ReceiveLastProcessMessage(i32),
    ProvideF32(String, f32),
}

fn get_string_from_caller(caller: &mut Caller<State>, ptr: i32, len: i32) -> String {
    use core::str::from_utf8;
    // Use our `caller` context to learn about the memory export of the
    // module which called this host function.
    let mem = caller.get_export("memory").unwrap();
    // Use the `ptr` and `len` values to get a subslice of the wasm-memory
    // which we'll attempt to interpret as utf-8.
    let store = &mut caller.as_context_mut();
    let ptr = ptr as u32 as usize;
    let len = len as u32 as usize;
    // println!("caller ptr: {}, len: {}", ptr, len);
    let data = mem.into_memory().unwrap().data(store).get(ptr..(ptr + len));
    let string = from_utf8(data.unwrap()).unwrap();
    string.to_string()
}

fn get_string_from_memory(
    memory: &Memory,
    store: &mut Store<State>,
    ptr: i32,
    len: i32,
) -> Option<String> {
    use core::str::from_utf8;
    let ptr = ptr as u32 as usize;
    let len = len as u32 as usize;
    let data = memory.data(store).get(ptr..(ptr + len));
    let string = from_utf8(data.unwrap());
    match string {
        Ok(string) => Some(string.to_string()),
        Err(err) => {
            println!("Error getting utf8 data: {:?}", err);
            None
        }
    }
}

struct WasmInstance {
    instance: Instance,
    store: Store<State>,
    engine: Arc<Engine>,
    linker: Linker<State>,
    path: String,
}

impl WasmInstance {
    async fn new(
        engine: Arc<Engine>,
        wasm_path: &str,
        sender: Sender<OutMessage>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = Dir::from_std_file(
            std::fs::File::open(Path::new(wasm_path).parent().unwrap()).unwrap(),
        );

        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_args()?
            .preopened_dir(dir, ".")?
            .build();

        let mut linker: Linker<State> = Linker::new(&engine);
        wasmtime_wasi::add_to_linker(&mut linker, |s| &mut s.wasi)?;
        Self::setup_host_functions(&mut linker)?;

        let mut store = Store::new(&engine, State::new(wasi, sender));
        let module = Module::from_file(&engine, wasm_path)?;

        let instance = linker.instantiate_async(&mut store, &module).await?;
        Ok(Self {
            instance,
            store,
            engine,
            linker,
            path: wasm_path.to_string(),
        })
    }

    async fn call_typed_func<Params, Results>(
        &mut self,
        name: &str,
        params: Params,
        deadline: u64,
    ) -> anyhow::Result<Results>
    where
        Params: WasmParams,
        Results: WasmResults,
    {
        self.store.epoch_deadline_async_yield_and_update(deadline);

        let func = self
            .instance
            .get_typed_func::<Params, Results>(&mut self.store, name)
            .unwrap();
        let result = func.call_async(&mut self.store, params).await?;
        Ok(result)
    }

    fn setup_host_functions(linker: &mut Linker<State>) -> Result<(), Box<dyn Error>> {
        linker.func_wrap(
            "host",
            "draw_rect",
            |mut caller: Caller<'_, State>, x: f32, y: f32, width: f32, height: f32| {
                let state = caller.data_mut();
                state.commands.push(Command::DrawRect(x, y, width, height));
            },
        )?;
        linker.func_wrap0_async(
            "host",
            "get_async_thing",
            |mut caller: Caller<'_, State>| {
                Box::new(async move {
                    let state = caller.data_mut();
                    let (sender, receiver) = oneshot::channel();
                    state.sender.start_send(OutMessage {
                        message_id: 0,
                        wasm_id: 0,
                        payload: OutPayload::NeededValue("hardcoded".to_string(), sender),
                    })?;
                    let result = receiver.await;
                    match result {
                        Ok(result) => {
                            println!("got result: {}", result);
                        }
                        Err(_) => {
                            println!("Cancelled")
                        }
                    }
                    let (ptr, _len) =
                        WasmInstance::transfer_string_to_wasm(&mut caller, "hardcoded".to_string())
                            .await
                            .unwrap();

                    Ok(ptr)
                })
            },
        )?;
        linker.func_wrap(
            "host",
            "draw_str",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32, x: f32, y: f32| {
                let string = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                state.commands.push(Command::DrawString(string, x, y));
            },
        )?;
        linker.func_wrap(
            "host",
            "provide_f32",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32, val: f32| {
                let string = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                state.commands.push(Command::ProvideF32(string, val));
            },
        )?;
        linker.func_wrap("host", "get_x", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.position.x
        })?;
        linker.func_wrap("host", "get_y", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.position.y
        })?;
        linker.func_wrap(
            "host",
            "clip_rect",
            |mut caller: Caller<'_, State>, x: f32, y: f32, width: f32, height: f32| {
                let state = caller.data_mut();
                state.commands.push(Command::ClipRect(x, y, width, height));
            },
        )?;
        linker.func_wrap(
            "host",
            "draw_rrect",
            |mut caller: Caller<'_, State>,
             x: f32,
             y: f32,
             width: f32,
             height: f32,
             radius: f32| {
                let state = caller.data_mut();
                state
                    .commands
                    .push(Command::DrawRRect(x, y, width, height, radius));
            },
        )?;
        linker.func_wrap(
            "host",
            "translate",
            |mut caller: Caller<'_, State>, x: f32, y: f32| {
                let state = caller.data_mut();
                state.commands.push(Command::Translate(x, y));
            },
        )?;
        linker.func_wrap("host", "save", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.commands.push(Command::Save);
        })?;
        linker.func_wrap("host", "restore", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.commands.push(Command::Restore);
        })?;
        linker.func_wrap(
            "host",
            "set_color",
            |mut caller: Caller<'_, State>, r: f32, g: f32, b: f32, a: f32| {
                let state = caller.data_mut();
                state.commands.push(Command::SetColor(r, g, b, a));
            },
        )?;

        linker.func_wrap(
            "host",
            "start_process_low_level",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32| -> u32 {
                let process = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                // TODO: Real process id
                let process_id = 0;
                state
                    .commands
                    .push(Command::StartProcess(process_id, process));
                process_id
            },
        )?;

        linker.func_wrap(
            "host",
            "set_get_state",
            |mut caller: Caller<'_, State>, ptr: u32, len: u32| {
                let state = caller.data_mut();
                state.get_state_info = (ptr, len);
            },
        )?;

        linker.func_wrap(
            "host",
            "send_message_low_level",
            |mut caller: Caller<'_, State>, process_id: i32, ptr: i32, len: i32| {
                let message = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                state
                    .commands
                    .push(Command::SendProcessMessage(process_id, message));
            },
        )?;

        linker.func_wrap(
            "host",
            "recieve_last_message_low_level",
            |mut caller: Caller<'_, State>, ptr: i32, process_id: i32| {
                {
                    let state = caller.data_mut();
                    state
                        .commands
                        .push(Command::ReceiveLastProcessMessage(process_id));
                }
                let state = caller.data_mut();
                let message = state
                    .process_messages
                    .get(&process_id)
                    .unwrap_or(&"test".to_string())
                    .clone();
                let message = message.as_bytes();
                let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
                // This is wrong. I need to figure out how I'm supposed to encode this stuff
                let store = caller.as_context_mut();
                memory.write(store, 0, message).unwrap();

                let mut bytes = [0u8; 8];
                bytes[0..4].copy_from_slice(&0_i32.to_le_bytes());
                bytes[4..8].copy_from_slice(&(message.len() as i32).to_le_bytes());

                let store = caller.as_context_mut();
                memory.write(store, ptr as usize, &bytes).unwrap();
            },
        )?;

        // TODO: Need to deal with paints

        Ok(())
    }

    pub async fn draw(&mut self, fn_name: &str) -> Result<Vec<Command>, Box<dyn Error>> {
        let _max_width = 0.0;
        let _max_height = 0.0;

        self.call_typed_func(fn_name, (), 1).await?;

        let state = &mut self.store.data_mut();

        let _paint = skia_safe::Paint::default();
        let commands = state.commands.clone();
        state.commands.clear();
        Ok(commands)
    }

    pub async fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_click", (x, y), 1)
            .await?;
        Ok(())
    }

    pub async fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f64, f64), ()>("on_scroll", (x, y), 1)
            .await?;
        Ok(())
    }

    pub async fn on_key(
        &mut self,
        key_code: u32,
        state: u32,
        modifiers: u32,
    ) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(u32, u32, u32), ()>("on_key", (key_code, state, modifiers), 1)
            .await?;

        Ok(())
    }

    pub async fn on_process_message(&mut self, process_id: i32, message: String) -> Result<(), Box<dyn Error>> {
        let (ptr, _len) = self.transfer_string_to_wasm2(message).await?;

        self.call_typed_func::<(i32, u32), ()>("on_process_message", (process_id, ptr), 1)
            .await?;
        Ok(())
    }

    pub async fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        if let Ok(json_string) = self.get_state().await.ok_or("no get state function") {
            let data = json_string.as_bytes();

            let module = Module::from_file(&self.engine, &self.path)?;
            let instance = self
                .linker
                .instantiate_async(&mut self.store, &module)
                .await?;
            self.instance = instance;
            self.set_state(data).await?;
        } else {
            let module = Module::from_file(&self.engine, &self.path)?;
            let instance = self
                .linker
                .instantiate_async(&mut self.store, &module)
                .await?;
            self.instance = instance;
        }

        Ok(())
    }

    pub async fn transfer_string_to_wasm(
        caller: &mut Caller<'_, State>,
        data: String,
    ) -> Result<(u32, u32), Box<dyn Error>> {
        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();

        let memory_size = (memory.data_size(caller.as_context_mut()) as f32
            / ByteSize::kb(64).as_u64() as f32)
            .ceil() as usize;

        let data_length_in_64k_multiples =
            (data.len() as f32 / ByteSize::kb(64).as_u64() as f32).ceil() as usize;
        if data_length_in_64k_multiples > memory_size {
            let delta = data_length_in_64k_multiples;
            memory
                .grow(caller.as_context_mut(), delta as u64 + 10)
                .unwrap();
        }

        let func = caller.get_export("alloc_string").unwrap();
        let func = func.into_func().unwrap();
        let results = &mut [Val::I32(0)];
        func.call_async(
            caller.as_context_mut(),
            &[Val::I32(data.len() as i32)],
            results,
        )
        .await
        .unwrap();
        let ptr = results[0].clone().i32().unwrap() as u32;

        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();

        memory
            .write(caller.as_context_mut(), ptr as usize, data.as_bytes())
            .unwrap();

        Ok((ptr, data.len() as u32))
    }


    // Instance vs caller. Can I collapse these?
    // super ugly that I right now have 3
    pub async fn transfer_string_to_wasm2(
        &mut self,
        data: String,
    ) -> Result<(u32, u32), Box<dyn Error>> {
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();

        let memory_size = (memory.data_size(&mut self.store) as f32
        / ByteSize::kb(64).as_u64() as f32)
        .ceil() as usize;

        let data_length_in_64k_multiples =
            (data.len() as f32 / ByteSize::kb(64).as_u64() as f32).ceil() as usize;
        if data_length_in_64k_multiples > memory_size {
            let delta = data_length_in_64k_multiples;
            memory.grow(&mut self.store, delta as u64 + 10).unwrap();
        }

        let ptr = self
            .call_typed_func::<u32, u32>("alloc_string", data.len() as u32, 1)
            .await?;
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();

        memory.write(&mut self.store, ptr as usize, data.as_bytes()).unwrap();

        Ok((ptr, data.len() as u32))
    }

    pub async fn set_state(&mut self, data: &[u8]) -> Result<(), Box<dyn Error>> {
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();

        let memory_size = (memory.data_size(&mut self.store) as f32
            / ByteSize::kb(64).as_u64() as f32)
            .ceil() as usize;

        let data_length_in_64k_multiples =
            (data.len() as f32 / ByteSize::kb(64).as_u64() as f32).ceil() as usize;
        if data_length_in_64k_multiples > memory_size {
            let delta = data_length_in_64k_multiples;
            memory.grow(&mut self.store, delta as u64 + 10).unwrap();
        }

        let ptr = self
            .call_typed_func::<u32, u32>("alloc_state", data.len() as u32, 1)
            .await?;
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();
        // let memory_size = memory.data_size(&mut self.store);

        memory.write(&mut self.store, ptr as usize, data).unwrap();

        self.call_typed_func::<(u32, u32), ()>("set_state", (ptr, data.len() as u32), 1)
            .await
            .unwrap();
        Ok(())
    }

    pub async fn get_state(&mut self) -> Option<String> {
        self.call_typed_func::<(), ()>("get_state", (), 1)
            .await
            .ok()?;
        let (ptr, len) = self.store.data().get_state_info;
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();

        let json_string = get_string_from_memory(&memory, &mut self.store, ptr as i32, len as i32);
        if json_string.is_none() {
            println!("No json string");
        }
        json_string
    }
}