use std::{
    collections::HashMap, error::Error, io::Write, path::Path, str::from_utf8, sync::Arc, thread,
    time::Duration,
};

use bytesize::ByteSize;

use framework::{CursorIcon, Position, Value};
use futures::{
    channel::{
        mpsc::{channel, Receiver, Sender},
        oneshot,
    },
    executor::{LocalPool, LocalSpawner},
    task::LocalSpawnExt,
    StreamExt,
};
use wasmtime::{
    AsContextMut, Caller, Config, Engine, Instance, Linker, Memory, Module, Store, Val, WasmParams,
    WasmResults,
};
use wasmtime_wasi::{Dir, WasiCtxBuilder};

use crate::{event::Event, keyboard::KeyboardInput};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct PointerLengthString {
    ptr: u32,
    len: u32,
}

pub type WasmId = u64;

#[derive(Debug, Clone)]
pub enum Payload {
    OnClick(Position),
    RunDraw(String),
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
    GetCommands,
    OnMove(f32, f32),
    NewSender(u32, u32, Sender<OutMessage>),
    OnDelete,
}

#[derive(Clone, Debug)]
pub struct Message {
    pub message_id: usize,
    pub external_id: Option<u32>,
    pub payload: Payload,
}

#[derive(Debug)]
pub enum OutPayload {
    DrawCommands(Vec<DrawCommands>),
    Saved(SaveState),
    ErrorPayload(String),
    Complete,
    NeededValue(String, oneshot::Sender<Value>),
    Error(String),
    Reloaded,
    Update(Vec<Commands>),
}

#[derive(Debug)]
pub struct OutMessage {
    pub message_id: usize,
    pub payload: OutPayload,
    pub external_id: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SaveState {
    Unsaved,
    Empty,
    Saved(serde_json::Value),
}

pub struct WasmMessenger {
    local_pool: futures::executor::LocalPool,
    local_spawner: LocalSpawner,
    last_wasm_id: u64,
    last_message_id: usize,
    engine: Arc<Engine>,
    senders: HashMap<WasmId, Sender<Message>>,
}

impl WasmMessenger {
    pub fn new() -> Self {
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
            last_message_id: 1,
            engine,
            senders: HashMap::new(),
        }
    }

    pub fn get_sender(&self, id: WasmId) -> Sender<Message> {
        self.senders.get(&id).unwrap().clone()
    }

    fn next_message_id(&mut self) -> usize {
        self.last_message_id += 1;
        self.last_message_id
    }

    fn next_wasm_id(&mut self) -> WasmId {
        self.last_wasm_id += 1;
        self.last_wasm_id
    }

    pub fn new_instance(
        &mut self,
        wasm_path: &str,
        partial_state: Option<String>,
        values: HashMap<String, Value>,
        external_sender: std::sync::mpsc::Sender<Event>,
        widget_id: usize,
    ) -> (WasmId, Receiver<OutMessage>) {
        let id = self.next_wasm_id();

        let (sender, receiver) = channel::<Message>(100000);
        let (out_sender, out_receiver) = channel::<OutMessage>(100000);

        let message_id = self.next_message_id();
        let message = Message {
            message_id,
            external_id: None,
            payload: Payload::PartialState(partial_state),
        };
        sender.clone().try_send(message).unwrap();
        // self.receivers.insert(id, out_receiver);
        self.senders.insert(id, sender);

        async fn spawn_instance(
            engine: Arc<Engine>,
            wasm_id: WasmId,
            wasm_path: String,
            receiver: Receiver<Message>,
            sender: Sender<OutMessage>,
            values: HashMap<String, Value>,
            external_sender: std::sync::mpsc::Sender<Event>,
            widget_id: usize,
        ) {
            let mut instance = WasmManager::new(
                engine.clone(),
                wasm_id,
                wasm_path.to_string(),
                receiver,
                sender,
                values,
                external_sender,
                widget_id,
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
                values,
                external_sender,
                widget_id,
            ))
            .unwrap();

        (id, out_receiver)
    }

    pub fn tick(&mut self) {
        // TODO: I don't think run_until or try_run_one
        // are the right options. I want run with max.
        // I think run_until literally keeps
        // trying to run even if there is on work.
        // use futures_timer::Delay;
        // self.local_pool
        //     .run_until(Delay::new(Duration::from_millis(8)));

        use std::time::Instant;
        let start_time = Instant::now();
        loop {
            if self.local_pool.try_run_one() {
                if start_time.elapsed().as_millis() > 8 {
                    break;
                }
            } else {
                break;
            }
        }
    }

    pub fn notify_external_sender(
        &mut self,
        wasm_id: usize,
        external_id: u32,
        widget_id: usize,
        sender: Sender<OutMessage>,
    ) {
        let mut wasm_sender = self.get_sender(wasm_id as u64);
        let message_id = self.next_message_id();
        wasm_sender
            .try_send(Message {
                message_id,
                external_id: Some(external_id),
                payload: Payload::NewSender(external_id, widget_id as u32, sender),
            })
            .unwrap();
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
    sender: Sender<OutMessage>,
    other_senders: HashMap<u32, (u32, Sender<OutMessage>)>,
}

impl WasmManager {
    pub async fn new(
        engine: Arc<Engine>,
        wasm_id: WasmId,
        wasm_path: String,
        receiver: Receiver<Message>,
        sender: Sender<OutMessage>,
        values: HashMap<String, Value>,
        external_sender: std::sync::mpsc::Sender<Event>,
        widget_id: usize,
    ) -> Self {
        let instance = WasmInstance::new(
            engine.clone(),
            &wasm_path,
            sender.clone(),
            wasm_id,
            values,
            external_sender,
            widget_id,
        )
        .await
        .unwrap();

        Self {
            id: wasm_id,
            instance,
            receiver,
            sender,
            other_senders: HashMap::new(),
        }
    }

    pub async fn init(&mut self) {
        loop {
            let message = self.receiver.select_next_some().await;
            let message_id = message.message_id;
            let external_id = message.external_id;
            // TODO: Can I wait for either this message or some reload message so I can kill infinite loops?
            let out_message = self.process_message(message).await;
            match out_message {
                Ok(out_messages) => {
                    for out_message in out_messages {
                        let result = if let Some(external_id) = out_message.external_id {
                            self.other_senders
                                .get_mut(&external_id)
                                .unwrap()
                                .1
                                .start_send(out_message)
                        } else {
                            self.sender.start_send(out_message)
                        };
                        if result.is_err() {
                            println!("Error sending message {:?}", result);
                        }
                    }
                }
                Err(err) => {
                    println!("Error processing message: {}", err);
                    let out_message = OutMessage {
                        message_id,
                        payload: OutPayload::Error(err.to_string()),
                        external_id,
                    };
                    let result = self.sender.start_send(out_message);
                    if result.is_err() {
                        println!("Error sending message {:?}", result);
                    }
                }
            }
        }
    }

    pub async fn process_message(
        &mut self,
        message: Message,
    ) -> Result<Vec<OutMessage>, Box<dyn Error>> {
        let default_return = Ok(vec![OutMessage {
            message_id: message.message_id,
            payload: OutPayload::Complete,
            external_id: message.external_id,
        }]);

        if let Some(external_id) = message.external_id {
            self.instance.set_widget_identifer(external_id).await?;
        }

        let result = match message.payload {
            Payload::OnClick(position) => {
                self.instance.on_click(position.x, position.y).await?;
                default_return
            }
            Payload::OnDelete => {
                self.instance.on_delete().await?;
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
            Payload::RunDraw(fn_name) => {
                let result = self.instance.draw(&fn_name).await;
                let mut commands_by_widget: HashMap<Option<u32>, Vec<DrawCommands>> =
                    HashMap::new();
                match result {
                    Ok(result) => {
                        let mut current_widget = message.external_id;
                        for command in result {
                            match command {
                                DrawCommands::ChangeWidget(widget_id) => {
                                    current_widget = Some(widget_id);
                                }
                                DrawCommands::DefaultWidget => {
                                    current_widget = None;
                                }
                                _ => {
                                    commands_by_widget
                                        .entry(current_widget)
                                        .or_insert_with(Vec::new)
                                        .push(command);
                                }
                            }
                        }
                        let mut out_messages = Vec::new();
                        for (widget_id, commands) in commands_by_widget {
                            out_messages.push(OutMessage {
                                message_id: message.message_id,
                                payload: OutPayload::DrawCommands(commands),
                                external_id: widget_id,
                            });
                        }
                        if out_messages.is_empty() {
                            return default_return;
                        }
                        Ok(out_messages)
                    }
                    Err(error) => {
                        println!("Error drawing {:?}", error);
                        default_return
                    }
                }
            }
            Payload::GetCommands => {
                let mut commands = self.instance.get_and_clear_commands();
                if commands.is_empty() {
                    return default_return;
                }

                for command in commands.iter_mut() {
                    match command {
                        Commands::MarkDirty(external_id) => {
                            if let Some((id, _)) = self.other_senders.get(external_id) {
                                println!("Swapping id {} for {}", external_id, id);
                                *external_id = *id;
                            } else {
                                println!("No external sender, delaying command");
                                self.instance.add_command(Commands::MarkDirty(*external_id));
                            }
                        }
                        _ => {}
                    }
                }
                Ok(vec![OutMessage {
                    message_id: message.message_id,
                    payload: OutPayload::Update(commands),
                    external_id: message.external_id,
                }])
            }
            Payload::OnScroll(x, y) => {
                self.instance.on_scroll(x, y).await?;
                default_return
            }
            Payload::ProcessMessage(process_id, message) => {
                self.instance
                    .on_process_message(process_id as u32, message)
                    .await?;
                default_return
            }
            Payload::OnKey(input) => {
                let (key_code, state, modifiers) = input.as_u32_tuple();
                let result = self.instance.on_key(key_code, state, modifiers).await;
                match result {
                    Ok(_) => default_return,
                    Err(err) => Ok(vec![OutMessage {
                        message_id: message.message_id,
                        payload: OutPayload::ErrorPayload(err.to_string()),
                        external_id: message.external_id,
                    }]),
                }
            }
            Payload::Reload => {
                match self.instance.reload().await {
                    Ok(_) => {}
                    Err(e) => {
                        println!("Error reloading {}", e);
                    }
                }
                Ok(vec![OutMessage {
                    message_id: message.message_id,
                    payload: OutPayload::Reloaded,
                    external_id: message.external_id,
                }])
            }
            Payload::SaveState => {
                let state = self.instance.get_state().await;
                match state {
                    Some(state) => {
                        if state.starts_with('\"') {
                            assert!(state.ends_with('\"'), "State is corrupt: {}", state);
                        }
                        Ok(vec![OutMessage {
                            message_id: message.message_id,
                            payload: OutPayload::Saved(SaveState::Saved(serde_json::from_str(
                                &state,
                            )?)),
                            external_id: message.external_id,
                        }])
                    }
                    None => {
                        println!("Failed to get state");
                        Ok(vec![OutMessage {
                            message_id: message.message_id,
                            payload: OutPayload::Saved(SaveState::Empty),
                            external_id: message.external_id,
                        }])
                    }
                }
            }
            Payload::PartialState(partial_state) => {
                self.instance
                    .set_state(partial_state.unwrap_or("{}".to_string()).as_bytes())
                    .await?;
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
            Payload::NewSender(external_id, widget_id, sender) => {
                self.other_senders.insert(external_id, (widget_id, sender));
                default_return
            }
        };

        if message.external_id.is_some() {
            self.instance.clear_widget_identifier().await?;
        }

        result
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
    values: HashMap<String, Value>,
    receivers: HashMap<String, oneshot::Receiver<Value>>,
    wasm_id: u64,
    widget_id: usize,
    external_id: Option<u32>,
    external_sender: std::sync::mpsc::Sender<Event>,
    process_id: u32,
}

impl State {
    fn new(
        wasi: wasmtime_wasi::WasiCtx,
        sender: Sender<OutMessage>,
        wasm_id: u64,
        values: HashMap<String, Value>,
        external_sender: std::sync::mpsc::Sender<Event>,
        widget_id: usize,
    ) -> Self {
        Self {
            wasi,
            draw_commands: Vec::new(),
            commands: Vec::new(),
            process_messages: HashMap::new(),
            get_state_info: (0, 0),
            position: Position { x: 0.0, y: 0.0 },
            sender,
            wasm_id,
            values,
            receivers: HashMap::new(),
            external_id: None,
            external_sender,
            process_id: 0,
            widget_id,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Commands {
    StartProcess(u32, String),
    SendProcessMessage(i32, String),
    ReceiveLastProcessMessage(i32),
    ProvideValue(String, Vec<u8>),
    Event(String, String),
    Subscribe(String),
    Unsubscribe(String),
    SetCursor(CursorIcon),
    Redraw(usize),
    CreateWidget(usize, f32, f32, f32, f32, u32),
    MarkDirty(u32),
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
    ChangeWidget(u32),
    DefaultWidget,
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
    widget_id: usize,
}

impl WasmInstance {
    async fn call_main(&mut self) -> anyhow::Result<()> {
        self.call_typed_func("main", self.widget_id as u32, 1)
            .await?;
        Ok(())
    }

    async fn new(
        engine: Arc<Engine>,
        wasm_path: &str,
        sender: Sender<OutMessage>,
        wasm_id: u64,
        values: HashMap<String, Value>,
        external_sender: std::sync::mpsc::Sender<Event>,
        widget_id: usize,
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

        let mut store = Store::new(
            &engine,
            State::new(wasi, sender, wasm_id, values, external_sender, widget_id),
        );
        let module = Module::from_file(&engine, wasm_path)?;

        let instance = linker.instantiate_async(&mut store, &module).await?;
        let mut me = Self {
            instance,
            store,
            engine,
            linker,
            path: wasm_path.to_string(),
            widget_id,
        };
        me.call_main().await?;
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
                    let (sender, receiver) = channel::<Vec<u8>>(1);
                    // Handle when it blocks and when it doesn't.
                    // Probably want a try_ version

                    state
                        .external_sender
                        .send(Event::ValueNeeded2(name.clone(), sender))?;
                    // The value is serialized and will need to be deserialized
                    let (result, _) = receiver.into_future().await;
                    match result {
                        Some(result) => {
                            state.values.insert(name, result.clone());
                            let (ptr, _len) = WasmInstance::transfer_string_to_wasm(
                                &mut caller,
                                from_utf8(&result).unwrap().to_string(),
                            )
                            .await
                            .unwrap();
                            Ok(ptr)
                        }
                        _ => Ok(0),
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

                    let mut should_remove = false;
                    if let Some(receiver) = state.receivers.get_mut(&name) {
                        if let Ok(Some(result)) = receiver.try_recv() {
                            state.values.insert(name.clone(), result);
                            should_remove = true;
                        }
                    }

                    if should_remove {
                        state.receivers.remove(&name);
                    }
                    // TODO: Unnecessary vec to string
                    if let Some(value) = state.values.get(&name) {
                        let value = value.clone();
                        let (ptr, _len) = WasmInstance::transfer_string_to_wasm(
                            &mut caller,
                            from_utf8(&value).unwrap().to_string(),
                        )
                        .await
                        .unwrap();
                        Ok(ptr)
                    } else {
                        let (sender, receiver) = oneshot::channel();
                        state.receivers.insert(name.clone(), receiver);
                        state.sender.start_send(OutMessage {
                            message_id: 0,
                            payload: OutPayload::NeededValue(name, sender),
                            external_id: state.external_id,
                        })?;

                        Ok(0)
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
            "send_event",
            |mut caller: Caller<'_, State>,
             kind_ptr: i32,
             kind_len: i32,
             event_ptr: i32,
             event_len: i32| {
                let kind = get_string_from_caller(&mut caller, kind_ptr, kind_len);
                let event = get_string_from_caller(&mut caller, event_ptr, event_len);
                let state = caller.data_mut();
                state
                    .external_sender
                    .send(Event::Event(kind, event))
                    .unwrap();
            },
        )?;

        linker.func_wrap(
            "host",
            "subscribe",
            |mut caller: Caller<'_, State>, kind_ptr: i32, kind_len: i32| {
                let kind = get_string_from_caller(&mut caller, kind_ptr, kind_len);
                let state = caller.data_mut();
                state
                    .external_sender
                    .send(Event::Subscribe(state.widget_id, kind))
                    .unwrap();
            },
        )?;

        linker.func_wrap(
            "host",
            "unsubscribe",
            |mut caller: Caller<'_, State>, kind_ptr: i32, kind_len: i32| {
                let kind = get_string_from_caller(&mut caller, kind_ptr, kind_len);
                let state = caller.data_mut();
                state
                    .external_sender
                    .send(Event::Unsubscribe(state.widget_id, kind))
                    .unwrap();
            },
        )?;

        linker.func_wrap(
            "host",
            "provide_value",
            |mut caller: Caller<'_, State>, name_ptr: i32, name_len: i32, ptr: i32, len: i32| {
                let string = get_string_from_caller(&mut caller, name_ptr, name_len);
                let data = get_bytes_from_caller(&mut caller, ptr, len).to_vec();
                let state = caller.data_mut();
                state
                    .external_sender
                    .send(Event::ProvideValue(string, data))
                    .unwrap();
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
            "change_widget",
            |mut caller: Caller<'_, State>, id: u32| {
                let state = caller.data_mut();
                state.draw_commands.push(DrawCommands::ChangeWidget(id));
            },
        )?;
        linker.func_wrap("host", "default_widget", |mut caller: Caller<'_, State>| {
            let state = caller.data_mut();
            state.draw_commands.push(DrawCommands::DefaultWidget);
        })?;
        linker.func_wrap(
            "host",
            "set_cursor_icon",
            |mut caller: Caller<'_, State>, cursor: u32| {
                let cursor_icon = CursorIcon::from(cursor);
                let state = caller.data_mut();
                state
                    .external_sender
                    .send(Event::SetCursor(cursor_icon))
                    .unwrap();
            },
        )?;

        linker.func_wrap(
            "host",
            "start_process_low_level",
            |mut caller: Caller<'_, State>, ptr: i32, len: i32| -> u32 {
                let process = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                // TODO: Real process id
                let process_id = state.process_id;
                state.process_id += 1;
                state
                    .external_sender
                    .send(Event::StartProcess(
                        process_id as usize,
                        state.widget_id,
                        process,
                    ))
                    .unwrap();
                process_id
            },
        )?;

        linker.func_wrap(
            "host",
            "mark_dirty",
            |mut caller: Caller<'_, State>, id: u32| {
                let state = caller.data_mut();

                state.external_sender.send(Event::MarkDirty(id)).unwrap();
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

        linker.func_wrap(
            "host",
            "create_widget",
            |mut caller: Caller<'_, State>,
             x: f32,
             y: f32,
             width: f32,
             height: f32,
             external_id: u32| {
                let state = caller.data_mut();
                state
                    .external_sender
                    .send(Event::CreateWidget(
                        state.wasm_id as usize,
                        x,
                        y,
                        width,
                        height,
                        external_id,
                    ))
                    .unwrap();
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
        let state = self.store.data_mut();
        let commands = state.commands.clone();
        state.commands.clear();
        commands
    }

    pub fn add_command(&mut self, command: Commands) {
        let state = self.store.data_mut();
        state.commands.push(command);
    }

    pub async fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_click", (x, y), 1)
            .await?;
        Ok(())
    }

    pub async fn on_delete(&mut self) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(), ()>("on_delete", (), 1).await?;
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
        process_id: u32,
        message: String,
    ) -> Result<(), Box<dyn Error>> {
        let (ptr, _len) = self.transfer_string_to_wasm2(message).await?;

        self.call_typed_func::<(u32, u32), ()>("on_process_message", (process_id, ptr), 1)
            .await?;
        Ok(())
    }

    pub async fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        if let Ok(json_string) = self.get_state().await.ok_or("no get state function") {
            let data = json_string.as_bytes();

            let context = self.get_context().await.ok_or("no get context function")?;

            let module = Module::from_file(&self.engine, &self.path)?;
            let instance = self
                .linker
                .instantiate_async(&mut self.store, &module)
                .await?;
            self.instance = instance;
            self.set_context(context.as_bytes()).await?;
            self.call_main().await?;
            self.set_state(data).await?;
        } else {
            let module = Module::from_file(&self.engine, &self.path)?;
            let instance = self
                .linker
                .instantiate_async(&mut self.store, &module)
                .await?;
            self.instance = instance;
            self.call_main().await?;
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

    pub async fn set_context(&mut self, data: &[u8]) -> Result<(), Box<dyn Error>> {
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

        self.call_typed_func::<(u32, u32), ()>("set_context", (ptr, data.len() as u32), 1)
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

    pub async fn get_context(&mut self) -> Option<String> {
        self.call_typed_func::<(), ()>("save_context", (), 1)
            .await
            .unwrap();
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
            .unwrap();
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

    pub async fn set_widget_identifer(&mut self, external_id: u32) -> Result<(), Box<dyn Error>> {
        let state = self.store.data_mut();
        state.external_id = Some(external_id);
        self.call_typed_func::<u32, ()>("set_widget_identifier", external_id, 1)
            .await?;
        Ok(())
    }

    pub async fn clear_widget_identifier(&mut self) -> Result<(), Box<dyn Error>> {
        let state = self.store.data_mut();
        state.external_id = None;
        self.call_typed_func::<(), ()>("clear_widget_identifier", (), 1)
            .await?;
        Ok(())
    }
}
