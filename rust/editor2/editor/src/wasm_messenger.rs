use std::{
    collections::{HashMap, HashSet},
    error::Error,
    io::Write,
    path::Path,
    sync::{mpsc, Arc},
    thread,
    time::Duration,
};

use bytesize::ByteSize;

use framework::CursorIcon;
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

use wasmtime::{
    AsContextMut, Caller, Config, Engine, Instance, Linker, Memory, Module, Store, Val, WasmParams,
    WasmResults,
};
use wasmtime_wasi::{Dir, WasiCtxBuilder};

use crate::{
    editor::Value,
    event::Event,
    keyboard::KeyboardInput,
    widget::Position,
    util::encode_base64,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct PointerLengthString {
    ptr: u32,
    len: u32,
}

pub type WasmId = u64;

#[derive(Debug, Clone)]
pub enum Payload {
    #[allow(unused)]
    NewInstance(String),
    OnClick(Position),
    Draw(String),
    OnScroll(f64, f64),
    OnKey(KeyboardInput),
    Reload,
    SaveState,
    ProcessMessage(usize, String),
    Event(String, String),
    OnSizeChange(f32, f32),
    OnMouseMove(Position, f32, f32),
    PartialState(Option<String>),
    OnMouseDown(Position),
    OnMouseUp(Position),
    Update,
    OnMove(f32, f32),
}

#[derive(Clone, Debug)]
pub struct Message {
    pub message_id: usize,
    pub wasm_id: WasmId,
    pub payload: Payload,
}

#[derive(Debug)]
pub enum OutPayload {
    DrawCommands(Vec<DrawCommands>),
    Saved(SaveState),
    ErrorPayload(String),
    Complete,
    NeededValue(String, oneshot::Sender<String>),
    Error(String),
    Reloaded,
    Update(Vec<Commands>),
}

#[derive(Debug)]
pub struct OutMessage {
    pub message_id: usize,
    pub wasm_id: WasmId,
    pub payload: OutPayload,
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
    wasm_draw_commands: HashMap<WasmId, Vec<DrawCommands>>,
    wasm_non_draw_commands: HashMap<WasmId, Vec<Commands>>,
    wasm_states: HashMap<WasmId, SaveState>,
    last_message_id: usize,
    // Not a huge fan of this solution,
    // but couldn't find a better way to dedup draws
    // Ideally, you can draw in the middle of click commands
    // I have some ideas.
    pending_messages: HashMap<WasmId, HashMap<usize, Message>>,
    engine: Arc<Engine>,
    receivers: HashMap<WasmId, Receiver<OutMessage>>,
    senders: HashMap<WasmId, Sender<Message>>,
    external_sender: Option<mpsc::Sender<Event>>,
    dirty_wasm: HashSet<WasmId>,
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
            last_message_id: 1,
            pending_messages: HashMap::new(),
            engine,
            receivers: HashMap::new(),
            senders: HashMap::new(),
            external_sender,
            dirty_wasm: HashSet::new(),
        }
    }

    pub fn get_and_drain_dirty_wasm(&mut self) -> HashSet<WasmId> {
        let mut dirty_wasm = HashSet::new();
        std::mem::swap(&mut dirty_wasm, &mut self.dirty_wasm);
        dirty_wasm
    }

    pub fn set_external_sender(&mut self, external_sender: mpsc::Sender<Event>) {
        self.external_sender = Some(external_sender);
    }

    pub fn number_of_pending_requests(&self) -> usize {
        let non_draw_commands_count = self
            .wasm_non_draw_commands
            .values()
            .map(|v| v.len())
            .sum::<usize>();
        let pending_message_count = self
            .pending_messages
            .values()
            .map(|v| v.len())
            .sum::<usize>();
        non_draw_commands_count + pending_message_count
    }

    pub fn get_sender(&self, id: WasmId) -> Sender<Message> {
        self.senders.get(&id).unwrap().clone()
    }

    pub fn pending_message_counts(&self) -> String {
        let mut stats: Vec<&str> = vec![];
        for messages_per in self.pending_messages.values() {
            for message in messages_per.values() {
                stats.push(match message.payload {
                    Payload::NewInstance(_) => "NewInstance",
                    Payload::OnClick(_) => "OnClick",
                    Payload::Draw(_) => "Draw",
                    Payload::OnScroll(_, _) => "OnScroll",
                    Payload::OnKey(_) => "OnKey",
                    Payload::Reload => "Reload",
                    Payload::SaveState => "SaveState",
                    Payload::ProcessMessage(_, _) => "ProcessMessage",
                    Payload::Event(_, _) => "Event",
                    Payload::OnSizeChange(_, _) => "OnSizeChange",
                    Payload::OnMouseMove(_, _, _) => "OnMouseMove",
                    Payload::PartialState(_) => "PartialState",
                    Payload::OnMouseDown(_) => "OnMouseDown",
                    Payload::OnMouseUp(_) => "OnMouseUp",
                    Payload::Update => "Update",
                    Payload::OnMove(_, _) => "OnMove",
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

    pub fn new_instance(&mut self, wasm_path: &str, partial_state: Option<String>) -> (WasmId, Receiver<OutMessage>) {
        let id = self.next_wasm_id();

        let (sender, receiver) = channel::<Message>(100000);
        let (out_sender, out_receiver) = channel::<OutMessage>(100000);

        // self.receivers.insert(id, out_receiver);
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

        let message_id = self.next_message_id();
        self.send_message(Message {
            message_id,
            wasm_id: id,
            payload: Payload::PartialState(partial_state),
        });

        (id, out_receiver)
    }

    pub fn process_non_draw_commands(&mut self, values: &mut HashMap<String, Value>) {
        for (wasm_id, commands) in self.wasm_non_draw_commands.iter() {
            for command in commands.iter() {
                match command {
                    Commands::StartProcess(process_id, process_command) => {
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
                    Commands::SendProcessMessage(process_id, message) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::SendProcessMessage(
                                *process_id as usize,
                                message.clone(),
                            ))
                            .unwrap();
                    }
                    Commands::ReceiveLastProcessMessage(_) => println!("Unhandled"),
                    Commands::ProvideF32(name, val) => {
                        values.insert(name.to_string(), Value::F32(*val));
                    }
                    Commands::ProvideBytes(name, data) => {
                        // TODO: Get rid of clone here
                        values.insert(name.to_string(), Value::Bytes(data.clone()));
                    }
                    Commands::Event(kind, event) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::Event(kind.clone(), event.clone()))
                            .unwrap();
                    }
                    Commands::Subscribe(kind) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::Subscribe(
                                // TODO: I probably actually want widget id?
                                *wasm_id as usize,
                                kind.clone(),
                            ))
                            .unwrap();
                    }
                    Commands::Unsubscribe(kind) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::Unsubscribe(
                                // TODO: I probably actually want widget id?
                                *wasm_id as usize,
                                kind.clone(),
                            ))
                            .unwrap();
                    }
                    Commands::Redraw => {
                        // TODO: Fix widget id once we move for widgets
                        // to handle this themselves
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::Redraw(0))
                            .unwrap();
                    }
                    Commands::SetCursor(cursor) => {
                        self.external_sender
                            .as_mut()
                            .unwrap()
                            .send(Event::SetCursor(*cursor))
                            .unwrap();
                    }
                }
            }
        }
        self.wasm_non_draw_commands.clear();
    }

    pub fn tick(&mut self, values: &mut HashMap<String, Value>) {
        self.process_non_draw_commands(values);
        // TODO: What is the right option here?
        self.local_pool
            .run_until(Delay::new(Duration::from_millis(4)));

        // I need to do this slightly differently because I need to draw in the context
        // of the widget.
        // But on tick I could get the pending drawings and then draw them
        // for each widget

        // TODO: need to time this out
        for out_receiver in self.receivers.values_mut() {
            while let Ok(Some(message)) = out_receiver.try_next() {
                // Note: Right now if a message doesn't have a corresponding in-message
                // I am just setting the out message to id: 0.
                if let Some(record) = self.pending_messages.get_mut(&message.wasm_id) {
                    record.remove(&message.message_id);
                } else {
                    println!("No pending message for {}", message.wasm_id)
                }

                // TODO: This just means we update everything every frame
                // Because the draw content might not have changed
                // but we still request it every frame. We should only request it
                // if things have actually changed
                // Or we should only consider it dirty if things changed.

                let mut should_mark_dirty = true;

                match message.payload {
                    OutPayload::DrawCommands(commands) => {
                        // TODO: Is this a performance issue?
                        // I'm thinking not? It seems to actually work because
                        // we only do this every once in a while
                        if self.wasm_draw_commands.get(&message.wasm_id) == Some(&commands) {
                            should_mark_dirty = false;
                        }
                        self.wasm_draw_commands.insert(message.wasm_id, commands);
                    }
                    OutPayload::Update(commands) => {
                        let current_commands = self
                            .wasm_non_draw_commands
                            .entry(message.wasm_id)
                            .or_default();
                        if current_commands.is_empty() {
                            should_mark_dirty = false;
                        }
                        current_commands.extend(commands);
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
                            // println!("Can't find value {}", name);
                        }
                        should_mark_dirty = false;
                    }
                    OutPayload::Reloaded => {
                        let commands = self
                            .wasm_non_draw_commands
                            .entry(message.wasm_id)
                            .or_default();
                        commands.push(Commands::Redraw);
                    }
                    OutPayload::Complete => {
                        // should_mark_dirty = false;
                    }
                    OutPayload::Error(error) => {
                        println!("Error: {}", error);
                    }
                }
                if should_mark_dirty {
                    self.dirty_wasm.insert(message.wasm_id);
                }
            }
        }
    }

    fn send_message(&mut self, message: Message) {
        let records = self.pending_messages.entry(message.wasm_id).or_default();

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

    pub fn has_draw_commands(&self, wasm_id: u64) -> bool {
        self.wasm_draw_commands
            .get(&wasm_id)
            .map(|x| !x.is_empty())
            .unwrap_or(false)
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
        let instance = WasmInstance::new(engine.clone(), &wasm_path, sender.clone(), wasm_id)
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
            let message_id = message.message_id;
            let wasm_id = self.id;
            // TODO: Can I wait for either this message or some reload message so I can kill infinite loops?
            let out_message = self.process_message(message).await;
            match out_message {
                Ok(out_message) => {
                    self.sender.start_send(out_message).unwrap();
                }
                Err(err) => {
                    println!("Error processing message: {}", err);
                    let out_message = OutMessage {
                        wasm_id,
                        message_id,
                        payload: OutPayload::Error(err.to_string()),
                    };
                    self.sender.start_send(out_message).unwrap();
                }
            }
        }
    }

    pub async fn process_message(
        &mut self,
        message: Message,
    ) -> Result<OutMessage, Box<dyn Error>> {
        let id = self.id;
        let default_return = Ok(OutMessage {
            wasm_id: self.id,
            message_id: message.message_id,
            payload: OutPayload::Complete,
        });

        match message.payload {
            Payload::NewInstance(_) => {
                panic!("Shouldn't get here")
            }
            Payload::OnClick(position) => {
                self.instance.on_click(position.x, position.y).await?;
                default_return
            }
            Payload::OnMouseDown(position) => {
                self.instance.on_mouse_down(position.x, position.y).await?;
                default_return
            }
            Payload::OnMouseUp(position) => {
                self.instance.on_mouse_up(position.x, position.y).await?;
                default_return
            }
            Payload::OnMouseMove(position, x_diff, y_diff) => {
                self.instance
                    .on_mouse_move(position.x, position.y, x_diff, y_diff)
                    .await?;
                default_return
            }
            Payload::Draw(fn_name) => {
                let result = self.instance.draw(&fn_name).await;
                match result {
                    Ok(result) => Ok(OutMessage {
                        message_id: message.message_id,
                        wasm_id: id,
                        payload: OutPayload::DrawCommands(result),
                    }),
                    Err(error) => {
                        println!("Error drawing {:?}", error);
                        default_return
                    }
                }
            }
            Payload::Update => {
                let commands = self.instance.get_and_clear_commands();
                Ok(OutMessage {
                    message_id: message.message_id,
                    wasm_id: id,
                    payload: OutPayload::Update(commands),
                })
            }
            Payload::OnScroll(x, y) => {
                self.instance.on_scroll(x, y).await?;
                default_return
            }
            Payload::ProcessMessage(process_id, message) => {
                self.instance
                    .on_process_message(process_id as i32, message)
                    .await?;
                default_return
            }
            Payload::OnKey(input) => {
                let (key_code, state, modifiers) = input.as_u32_tuple();
                let result = self.instance.on_key(key_code, state, modifiers).await;
                match result {
                    Ok(_) => default_return,
                    Err(err) => Ok(OutMessage {
                        wasm_id: self.id,
                        message_id: message.message_id,
                        payload: OutPayload::ErrorPayload(err.to_string()),
                    }),
                }
            }
            Payload::Reload => {
                match self.instance.reload().await {
                    Ok(_) => {}
                    Err(e) => {
                        println!("Error reloading {}", e);
                    }
                }
                Ok(OutMessage {
                    message_id: message.message_id,
                    wasm_id: id,
                    payload: OutPayload::Reloaded,
                })
            }
            Payload::SaveState => {
                let state = self.instance.get_state().await;
                match state {
                    Some(state) => {
                        if state.starts_with('\"') {
                            assert!(state.ends_with('\"'), "State is corrupt: {}", state);
                        }
                        Ok(OutMessage {
                            message_id: message.message_id,
                            wasm_id: id,
                            payload: OutPayload::Saved(SaveState::Saved(state)),
                        })
                    }
                    None => {
                        println!("Failed to get state");
                        Ok(OutMessage {
                            message_id: message.message_id,
                            wasm_id: id,
                            payload: OutPayload::Saved(SaveState::Empty),
                        })
                    }
                }
            }
            Payload::PartialState(partial_state) => {
                let encoded_state = encode_base64(&partial_state.unwrap_or("{}".to_string()));
                self.instance.set_state(encoded_state.as_bytes()).await?;
                // let state = self.instance.get_state().await;
                // if let Some(state) = state {
                //     let base64_decoded = decode_base64(&state.as_bytes().to_vec())?;
                //     let state = String::from_utf8(base64_decoded)?;
                //     let merged_state = merge_json(partial_state, state);
                //     let encoded_state = encode_base64(&merged_state);
                //     self.instance.set_state(encoded_state.as_bytes()).await?;
                // }
                default_return
            }
            Payload::OnMove(x, y) => {
                self.instance.store.data_mut().position = Position { x, y };
                self.instance.on_move(x, y).await?;
                default_return
            }
            Payload::Event(kind, event) => {
                self.instance.on_event(kind, event).await?;
                default_return
            }
            Payload::OnSizeChange(width, height) => {
                self.instance.on_size_change(width, height).await?;
                default_return
            }
        }
    }
}

struct State {
    wasi: wasmtime_wasi::WasiCtx,
    draw_commands: Vec<DrawCommands>,
    // TODO: Make separate type
    commands: Vec<Commands>,
    get_state_info: (u32, u32),
    // Probably not the best structure
    // but lets start here
    process_messages: HashMap<i32, String>,
    position: Position,
    sender: Sender<OutMessage>,
    wasm_id: u64,
}

impl State {
    fn new(wasi: wasmtime_wasi::WasiCtx, sender: Sender<OutMessage>, wasm_id: u64) -> Self {
        Self {
            wasi,
            draw_commands: Vec::new(),
            commands: Vec::new(),
            process_messages: HashMap::new(),
            get_state_info: (0, 0),
            position: Position { x: 0.0, y: 0.0 },
            sender,
            wasm_id,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Commands {
    StartProcess(u32, String),
    SendProcessMessage(i32, String),
    ReceiveLastProcessMessage(i32),
    ProvideF32(String, f32),
    ProvideBytes(String, Vec<u8>),
    Event(String, String),
    Subscribe(String),
    Unsubscribe(String),
    SetCursor(CursorIcon),
    Redraw,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DrawCommands {
    DrawRect(f32, f32, f32, f32),
    DrawString(String, f32, f32),
    ClipRect(f32, f32, f32, f32),
    DrawRRect(f32, f32, f32, f32, f32),
    Translate(f32, f32),
    SetColor(f32, f32, f32, f32),
    Restore,
    Save,
}

fn get_bytes_from_caller(caller: &mut Caller<State>, ptr: i32, len: i32) -> Vec<u8> {
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
    data.unwrap().to_vec()
}

fn get_string_from_caller(caller: &mut Caller<State>, ptr: i32, len: i32) -> String {
    use core::str::from_utf8;
    // I allocate a vector here I didn't need to for code reuse sake.
    // There is probably a better way to do this.
    let data = get_bytes_from_caller(caller, ptr, len);
    let string = from_utf8(&data).unwrap();
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
        wasm_id: u64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let dir = Dir::from_std_file(
            std::fs::File::open(Path::new(wasm_path).parent().unwrap()).unwrap(),
        );

        let code_dir = Dir::from_std_file(
            std::fs::File::open("/Users/jimmyhmiller/Documents/Code/PlayGround/rust/editor2")
                .unwrap(),
        );

        let vs_code_extension_dir = Dir::from_std_file(
            std::fs::File::open("/Users/jimmyhmiller/.vscode/extensions/").unwrap(),
        );

        let root_dir = Dir::from_std_file(std::fs::File::open("/").unwrap());

        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_args()?
            .preopened_dir(dir, ".")?
            .preopened_dir(code_dir, "/code")?
            .preopened_dir(root_dir, "/")?
            // TODO: How do we handle this in the general case?
            .preopened_dir(
                vs_code_extension_dir,
                "/Users/jimmyhmiller/.vscode/extensions/",
            )?
            .build();

        let mut linker: Linker<State> = Linker::new(&engine);
        wasmtime_wasi::add_to_linker(&mut linker, |s| &mut s.wasi)?;
        Self::setup_host_functions(&mut linker)?;

        let mut store = Store::new(&engine, State::new(wasi, sender, wasm_id));
        let module = Module::from_file(&engine, wasm_path)?;

        let instance = linker.instantiate_async(&mut store, &module).await?;
        let mut me = Self {
            instance,
            store,
            engine,
            linker,
            path: wasm_path.to_string(),
        };
        me.call_typed_func("main", (), 1).await?;
        Ok(me)
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
            .get_typed_func::<Params, Results>(&mut self.store, name)?;
        let result = func.call_async(&mut self.store, params).await?;
        Ok(result)
    }

    fn setup_host_functions(linker: &mut Linker<State>) -> Result<(), Box<dyn Error>> {
        linker.func_wrap(
            "host",
            "draw_rect",
            |mut caller: Caller<'_, State>, x: f32, y: f32, width: f32, height: f32| {
                let state = caller.data_mut();
                state
                    .draw_commands
                    .push(DrawCommands::DrawRect(x, y, width, height));
            },
        )?;
        linker.func_wrap2_async(
            "host",
            "get_value",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32| {
                let name = get_string_from_caller(&mut caller, ptr, len);
                Box::new(async move {
                    let state = caller.data_mut();
                    let (sender, receiver) = oneshot::channel();
                    // Handle when it blocks and when it doesn't.
                    // Probably want a try_ version
                    state.sender.start_send(OutMessage {
                        message_id: 0,
                        wasm_id: state.wasm_id,
                        payload: OutPayload::NeededValue(name, sender),
                    })?;
                    // The value is serialized and will need to be deserialized
                    let result = receiver.await;
                    match result {
                        Ok(result) => {
                            let (ptr, _len) =
                                WasmInstance::transfer_string_to_wasm(&mut caller, result)
                                    .await
                                    .unwrap();
                            Ok(ptr)
                        }
                        Err(_e) => {
                            // TODO: Actually handle
                            // println!("Cancelled");
                            Ok(0)
                        }
                    }
                })
            },
        )?;
        linker.func_wrap2_async(
            "host",
            "try_get_value",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32| {
                let name = get_string_from_caller(&mut caller, ptr, len);
                Box::new(async move {
                    let state = caller.data_mut();
                    let (sender, mut receiver) = oneshot::channel();
                    // Handle when it blocks and when it doesn't.
                    // Probably want a try_ version
                    state.sender.start_send(OutMessage {
                        message_id: 0,
                        wasm_id: state.wasm_id,
                        payload: OutPayload::NeededValue(name, sender),
                    })?;

                    // TODO: This will probably cause problems for the sender
                    let result = receiver.try_recv();
                    if result.is_err() {
                        return Ok(0);
                    }
                    let result = result.unwrap();
                    match result {
                        Some(result) => {
                            let (ptr, _len) =
                                WasmInstance::transfer_string_to_wasm(&mut caller, result)
                                    .await
                                    .unwrap();
                            Ok(ptr)
                        }
                        None => {
                            // TODO: Actually handle
                            println!("Cancelled");
                            Ok(0)
                        }
                    }
                })
            },
        )?;
        linker.func_wrap(
            "host",
            "draw_str",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32, x: f32, y: f32| {
                let string = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                state
                    .draw_commands
                    .push(DrawCommands::DrawString(string, x, y));
            },
        )?;
        linker.func_wrap(
            "host",
            "provide_f32",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32, val: f32| {
                let string = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                state.commands.push(Commands::ProvideF32(string, val));
            },
        )?;
        linker.func_wrap(
            "host",
            "send_event",
            |mut caller: Caller<'_, State>,
             kind_ptr: i32,
             kind_len: i32,
             event_ptr: i32,
             event_len: i32| {
                let kind = get_string_from_caller(&mut caller, kind_ptr, kind_len);
                let event = get_string_from_caller(&mut caller, event_ptr, event_len);
                let state = caller.data_mut();
                state.commands.push(Commands::Event(kind, event));
            },
        )?;

        linker.func_wrap(
            "host",
            "subscribe",
            |mut caller: Caller<'_, State>, kind_ptr: i32, kind_len: i32| {
                let kind = get_string_from_caller(&mut caller, kind_ptr, kind_len);
                let state = caller.data_mut();
                state.commands.push(Commands::Subscribe(kind));
            },
        )?;

        linker.func_wrap(
            "host",
            "unsubscribe",
            |mut caller: Caller<'_, State>, kind_ptr: i32, kind_len: i32| {
                let kind = get_string_from_caller(&mut caller, kind_ptr, kind_len);
                let state = caller.data_mut();
                state.commands.push(Commands::Unsubscribe(kind));
            },
        )?;

        linker.func_wrap(
            "host",
            "provide_bytes",
            |mut caller: Caller<'_, State>, name_ptr: i32, name_len: i32, ptr: i32, len: i32| {
                let string = get_string_from_caller(&mut caller, name_ptr, name_len);
                let data = get_bytes_from_caller(&mut caller, ptr, len).to_vec();
                let state = caller.data_mut();
                state.commands.push(Commands::ProvideBytes(string, data));
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
                state
                    .draw_commands
                    .push(DrawCommands::ClipRect(x, y, width, height));
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
                    .draw_commands
                    .push(DrawCommands::DrawRRect(x, y, width, height, radius));
            },
        )?;
        linker.func_wrap(
            "host",
            "translate",
            |mut caller: Caller<'_, State>, x: f32, y: f32| {
                let state = caller.data_mut();
                state.draw_commands.push(DrawCommands::Translate(x, y));
            },
        )?;
        linker.func_wrap("host", "save", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.draw_commands.push(DrawCommands::Save);
        })?;
        linker.func_wrap("host", "restore", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.draw_commands.push(DrawCommands::Restore);
        })?;
        linker.func_wrap(
            "host",
            "set_color",
            |mut caller: Caller<'_, State>, r: f32, g: f32, b: f32, a: f32| {
                let state = caller.data_mut();
                state.draw_commands.push(DrawCommands::SetColor(r, g, b, a));
            },
        )?;
        linker.func_wrap(
            "host",
            "set_cursor_icon",
            |mut caller: Caller<'_, State>, cursor: u32| {
                let cursor_icon = CursorIcon::from(cursor);
                let state = caller.data_mut();
                state.commands.push(Commands::SetCursor(cursor_icon));
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
                    .push(Commands::StartProcess(process_id, process));
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
                    .push(Commands::SendProcessMessage(process_id, message));
            },
        )?;

        linker.func_wrap(
            "host",
            "save_file_low_level",
            |mut caller: Caller<'_, State>,
             path_ptr: i32,
             path_len: i32,
             text_ptr: i32,
             text_len: i32| {
                let path = get_string_from_caller(&mut caller, path_ptr, path_len);
                let text = get_string_from_caller(&mut caller, text_ptr, text_len);
                // open file at path and save text
                // I should almost certainly do a command here but I just want to get it working
                let mut file = std::fs::File::create(path).unwrap();
                file.write_all(text.as_bytes()).unwrap();
            },
        )?;

        linker.func_wrap(
            "host",
            "receive_last_message_low_level",
            |mut caller: Caller<'_, State>, ptr: i32, process_id: i32| {
                {
                    let state = caller.data_mut();
                    state
                        .commands
                        .push(Commands::ReceiveLastProcessMessage(process_id));
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

    pub async fn draw(&mut self, fn_name: &str) -> Result<Vec<DrawCommands>, Box<dyn Error>> {
        let _max_width = 0.0;
        let _max_height = 0.0;

        self.call_typed_func(fn_name, (), 1).await?;

        let state = &mut self.store.data_mut();

        let _paint = skia_safe::Paint::default();
        let commands = state.draw_commands.clone();
        state.draw_commands.clear();
        Ok(commands)
    }

    pub fn get_and_clear_commands(&mut self) -> Vec<Commands> {
        let state = &mut self.store.data_mut();
        let commands = state.commands.clone();
        state.commands.clear();
        commands
    }

    pub async fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_click", (x, y), 1)
            .await?;
        Ok(())
    }

    async fn on_mouse_down(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_mouse_down", (x, y), 1)
            .await?;
        Ok(())
    }

    async fn on_mouse_up(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_mouse_up", (x, y), 1)
            .await?;
        Ok(())
    }

    pub async fn on_mouse_move(
        &mut self,
        x: f32,
        y: f32,
        x_diff: f32,
        y_diff: f32,
    ) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32, f32, f32), ()>(
            "on_mouse_move",
            (x, y, x_diff, y_diff),
            1,
        )
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

    pub async fn on_process_message(
        &mut self,
        process_id: i32,
        message: String,
    ) -> Result<(), Box<dyn Error>> {
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
            self.call_typed_func("main", (), 1).await?;
            self.set_state(data).await?;
        } else {
            let module = Module::from_file(&self.engine, &self.path)?;
            let instance = self
                .linker
                .instantiate_async(&mut self.store, &module)
                .await?;
            self.instance = instance;
            self.call_typed_func("main", (), 1).await?;
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

        memory
            .write(&mut self.store, ptr as usize, data.as_bytes())
            .unwrap();

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
            println!("Growing memory by {}", delta);
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
            println!("No json string {:?}", self.path);
        }
        self.call_typed_func::<(u32, u32), ()>("finish_get_state", (ptr, len), 1)
            .await
            .ok()?;
        json_string
    }

    pub async fn on_event(&mut self, kind: String, event: String) -> Result<(), Box<dyn Error>> {
        let (kind_ptr, _len) = self.transfer_string_to_wasm2(kind).await?;
        let (event_ptr, _len) = self.transfer_string_to_wasm2(event).await?;
        self.call_typed_func::<(u32, u32), ()>("on_event", (kind_ptr, event_ptr), 1)
            .await?;
        Ok(())
    }

    pub async fn on_size_change(&mut self, width: f32, height: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_size_change", (width, height), 1)
            .await?;
        Ok(())
    }

    pub async fn on_move(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_move", (x, y), 1)
            .await?;
        Ok(())
    }
}
