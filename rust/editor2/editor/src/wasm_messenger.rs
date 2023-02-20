use std::{sync::{Arc}, collections::HashMap, time::Duration, thread, error::Error};


use futures::{executor::LocalPool, task::LocalSpawnExt, channel::mpsc::{channel, Receiver, Sender}, StreamExt};
use futures::task::SpawnExt;
use futures_timer::Delay;
use skia_safe::{Canvas, Font, Typeface, FontStyle};
use wasmtime::{Engine, Config, Instance, Store, Linker, Module, WasmParams, WasmResults, Caller, Memory, AsContextMut};

use crate::widget::{Wasm, Size, Color};


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct PointerLengthString {
    ptr: u32,
    len: u32,
}

pub type WasmId = u64;

enum Payload {
    Noop,
    NewInstance(String),
    OnClick,
    Draw(String),
}

struct Message {
    wasm_id: WasmId,
    payload: Payload,
}

enum OutPayload {
    DrawCommands(Vec<Command>),
}

struct OutMessage {
    wasm_id: WasmId,
    payload: OutPayload,
}

pub struct WasmMessenger {
    sender: Sender<Message>,
    local_pool: futures::executor::LocalPool,
    last_wasm_id: u64,
    out_receiver: Receiver<OutMessage>,
    wasm_draw_commands: HashMap<WasmId, Vec<Command>>,
}

impl WasmMessenger {

    pub fn new() -> Self {
        let (sender, receiver) = channel::<Message>(1000);
        let (out_sender, out_receiver) = channel::<OutMessage>(1000);
        let local_pool = LocalPool::new();
        let local_spawner = local_pool.spawner();

        async fn init_manager(receiver: Receiver<Message>, out_sender: Sender<OutMessage>) {
            let mut wasm_manager = WasmManager::new(receiver, out_sender);
            wasm_manager.init().await
        }


        local_spawner.spawn_local(init_manager(receiver, out_sender)).unwrap();
        Self {
            sender,
            out_receiver,
            local_pool,
            last_wasm_id: 0,
            wasm_draw_commands: HashMap::new(),
        }

    }

    fn next_wasm_id(&mut self) -> WasmId {
        self.last_wasm_id += 1;
        self.last_wasm_id
    }

    pub fn spawn_wasm(&mut self, wasm_path: &str) -> WasmId {
        let id = self.next_wasm_id();
        self.sender.start_send(Message { wasm_id: id, payload: Payload::NewInstance(wasm_path.to_string()) }).unwrap();
        id
    }

    pub fn draw_widget(&mut self, wasm_id: WasmId, canvas: &mut Canvas) {
        if let Some(commands) = self.wasm_draw_commands.get(&wasm_id) {

            let mut max_width = 0.0;
            let mut max_height = 0.0;

            let mut paint = skia_safe::Paint::default();
            for command in commands.iter() {
                match command {
                    Command::SetColor(r, g, b, a) => {
                        paint.set_color(Color::new(*r, *g, *b, *a).to_color4f().to_color());
                    }
                    Command::DrawRect(x, y, width, height) => {
                        canvas.draw_rect(skia_safe::Rect::from_xywh(*x, *y, *width, *height), &paint);
                        // This is not quite right because of translate and stuff.
                        if *x + *width > max_width {
                            max_width = *x + *width;
                        }
                        if *y + *height > max_height {
                            max_height = *y + *height;
                        }
                    }
                    Command::DrawString(str, x, y) => {
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
                        if *width > max_width {
                            max_width = *width;
                        }
                        if *height > max_height {
                            max_height = *height;
                        }
                    }
                    Command::DrawRRect(x, y, width, height, radius) => {
                        let rrect = skia_safe::RRect::new_rect_xy(
                            skia_safe::Rect::from_xywh(*x, *y, *width, *height),
                            *radius,
                            *radius,
                        );
                        canvas.draw_rrect(&rrect, &paint);
                    }
                    Command::Translate(x, y) => {
                        canvas.translate((*x, *y));
                    }
                    Command::Save => {
                        canvas.save();
                    }
                    Command::Restore => {
                        canvas.restore();
                    }
                    c => {
                        // Need to move things out of draw
                        println!("Unknown command {:?}", c);
                    }
                }
            }
        }
    }

    pub fn tick(&mut self) {
        // Need a timeout instead
        self.local_pool.run_until(Delay::new(Duration::from_millis(1)));


        // I need to do this slightly differently because I need to draw in the context
        // of the widget.
        // But on tick I could get the pending drawings and then draw them
        // for each widget


        // TODO: need to time this out
        while let Ok(Some(message)) = self.out_receiver.try_next() {
            match message.payload {
                OutPayload::DrawCommands(commands) => {
                    self.wasm_draw_commands.insert(message.wasm_id, commands);
                }
            }
        }
    }

    pub fn send_noop(&mut self, wasm_id: WasmId) {
        self.sender.start_send(Message { wasm_id, payload: Payload::Noop}).unwrap();
    }

    pub fn send_on_click(&mut self, wasm_id: WasmId) {
        self.sender.start_send(Message { wasm_id, payload: Payload::OnClick}).unwrap();
    }

    pub fn send_draw(&mut self, wasm_id: WasmId, fn_name: &str) {
        self.sender.start_send(Message { wasm_id, payload: Payload::Draw(fn_name.to_string())}).unwrap();
    }

}



struct WasmManager {
    wasm_instances: HashMap<WasmId, WasmInstance>,
    receiver: Receiver<Message>,
    engine: Arc<Engine>,
    sender: Sender<OutMessage>
}


impl WasmManager {

    pub fn new(receiver: Receiver<Message>, sender: Sender<OutMessage>) -> Self {
        let mut config = Config::new();
        config.epoch_interruption(true);
        config.async_support(true);
        let engine = Arc::new(Engine::new(&config).unwrap());

        let engine_clone = engine.clone();
        thread::spawn(move || {
            loop {
                engine_clone.increment_epoch();
                thread::sleep(Duration::from_millis(1));
            }
        });

        Self {
            wasm_instances: HashMap::new(),
            receiver,
            engine,
            sender,
        }
    }

    pub async fn init(&mut self) {
        loop {
            println!("Waiting for message");
            self.process_message().await;
            println!("processed");
        }
    }

    pub async fn new_instance(&mut self, wasm_id: WasmId, wasm_path: &str) {
        self.wasm_instances.insert(wasm_id, WasmInstance::new(self.engine.clone(), wasm_path).await.unwrap());
    }

    pub async fn process_message(&mut self) {
        if let Some(message) = self.receiver.next().await {
            let id = message.wasm_id;
            match message.payload {
                Payload::Noop => {
                    println!("Got noop {}", id);
                    if let Some(instance) = self.wasm_instances.get(&id) {
                        instance.fake_invoke().await;
                    } else {
                        println!("Not found {}", id);
                    }
                }
                Payload::NewInstance(path) => {
                    self.new_instance(id, &path).await
                }
                Payload::OnClick => {
                    if let Some(instance) = self.wasm_instances.get_mut(&id) {
                        instance.on_click(0.0, 0.0).await.unwrap();
                    } else {
                        println!("Not found {}", id);
                    }
                }
                Payload::Draw(fn_name) => {
                    if let Some(instance) = self.wasm_instances.get_mut(&id) {
                        let result = instance.draw(&fn_name).await.unwrap();
                        self.sender.start_send(OutMessage {wasm_id: id, payload: OutPayload::DrawCommands(result)}).unwrap();
                    } else {
                        println!("Not found {}", id);
                    }
                }
            }
        }
    }
}


struct State {
    wasi: wasmtime_wasi::WasiCtx,
    commands: Vec<Command>,
    // Probably not the best structure
    // but lets start here
    process_messages: HashMap<i32, String>,
}

impl State {
    fn new(wasi: wasmtime_wasi::WasiCtx) -> Self {
        Self {
            wasi,
            commands: Vec::new(),
            process_messages: HashMap::new(),
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
    StartProcess(String),
    SendProcessMessage(i32, String),
    ReceiveLastProcessMessage(i32),
}


fn get_string_from_caller(caller: &mut Caller<State>, ptr: i32, len: i32) -> String {
    use core::str::from_utf8;
    // Use our `caller` context to learn about the memory export of the
    // module which called this host function.
    let mem = caller.get_export("memory").unwrap();
    // Use the `ptr` and `len` values to get a subslice of the wasm-memory
    // which we'll attempt to interpret as utf-8.
    let store = &mut caller.as_context_mut();
    let data = mem
        .into_memory()
        .unwrap()
        .data(store)
        .get(ptr as u32 as usize..)
        .and_then(|arr| arr.get(..len as u32 as usize));
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
    let data = memory
        .data(store)
        .get(ptr as u32 as usize..)
        .and_then(|arr| arr.get(..len as u32 as usize));
    let string = from_utf8(data.unwrap()).ok()?;
    Some(string.to_string())
}

struct WasmInstance {
    instance: Instance,
    store: Store<State>,
    engine: Arc<Engine>,
    linker: Linker<State>,
    path: String,
}


impl WasmInstance {

    async fn new(engine: Arc<Engine>, wasm_path: &str) -> Result<Self, Box<dyn std::error::Error>>{

        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_stderr()
            .inherit_env()?
            .build();

        let mut linker: Linker<State> = Linker::new(&engine);
        wasmtime_wasi::add_to_linker(&mut linker, |s| &mut s.wasi)?;
        Self::setup_host_functions(&mut linker)?;

        let mut store = Store::new(&engine, State::new(wasi));
        let module = Module::from_file(&engine, wasm_path)?;

        let instance = linker.instantiate_async(&mut store, &module).await?;
        Ok(Self { instance, store, engine, linker, path: wasm_path.to_string() })
    }

    async fn init(&mut self, label: i32) -> Result<(), Box<dyn std::error::Error>> {
        self.store.epoch_deadline_async_yield_and_update(1);
        let init = self.instance.get_typed_func::<i32, i32>(&mut self.store, "loop_forever")?;
        let result = init.call_async(&mut self.store, label).await?;
        println!("Result: {}", result);
        Ok(())
    }

    pub async fn fake_invoke(&self) {
        println!("Fake Invoked!");
    }

    async fn call_typed_func<Params, Results>(
        &mut self,
        name: &str,
        params: Params,
        deadline: u64,
    ) -> anyhow::Result<Results>
        where
        Params: WasmParams,
        Results: WasmResults
         {
        self.store.epoch_deadline_async_yield_and_update(deadline * 10);

        let func = self.instance.get_typed_func::<Params, Results>(&mut self.store, name)?;
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
            |mut caller: Caller<'_, State>, ptr: i32, len: i32| -> i32 {
                let process = get_string_from_caller(&mut caller, ptr, len);
                let state = caller.data_mut();
                state.commands.push(Command::StartProcess(process));
                0
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
                bytes[0..4].copy_from_slice(&(0 as i32).to_le_bytes());
                bytes[4..8].copy_from_slice(&(message.len() as i32).to_le_bytes());

                let store = caller.as_context_mut();
                memory.write(store, ptr as usize, &bytes).unwrap();
            },
        )?;

        // TODO: Need to deal with paints

        Ok(())
    }

    pub async fn draw(&mut self, fn_name: &str) -> Result<Vec<Command>, Box<dyn Error>> {


        let mut max_width = 0.0;
        let mut max_height = 0.0;

        self.call_typed_func(fn_name, (), 1).await?;

        let state = &mut self.store.data_mut();

        let mut paint = skia_safe::Paint::default();
        let commands = state.commands.clone();
        state.commands.clear();
        return Ok(commands);

    }

    pub async fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f32, f32), ()>("on_click", (x, y), 1).await?;
        Ok(())
    }

    pub async fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        self.call_typed_func::<(f64, f64), ()>("on_scroll", (x, y), 1).await?;
        Ok(())
    }

    pub async fn on_key(
        &mut self,
        key_code: u32,
        state: u32,
        modifiers: u32,
    ) -> Result<(), Box<dyn Error>> {

        self.call_typed_func::<(u32, u32, u32), ()>("on_key", (key_code, state, modifiers), 1).await?;

        Ok(())
    }


    pub async fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        let json_string = self.get_state().await.ok_or("no get state function")?;
        let data = json_string.as_bytes();

        let module = Module::from_file(&self.engine, &self.path)?;
        let instance = self.linker.instantiate(&mut self.store, &module)?;
        self.instance = instance;
        self.set_state(data).await?;

        Ok(())
    }

    pub async fn set_state(&mut self, data: &[u8]) -> Result<(), Box<dyn Error>> {
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();

        let memory_size = memory.data_size(&mut self.store);

        let data_length_in_64k_multiples = (data.len() as f32 / 65536.0).ceil() as usize;
        if data_length_in_64k_multiples > memory_size {
            let delta = data_length_in_64k_multiples - memory_size;
            memory.grow(&mut self.store, delta as u64 + 10).unwrap();
        }
        memory.write(&mut self.store, 0, &data).unwrap();

        self.call_typed_func::<(i32, i32), ()>("set_state", (0, data.len() as i32), 1).await?;
        Ok(())
    }

    pub async fn get_state(&mut self) -> Option<String> {
        let json_string_ptr = self.call_typed_func::<(), i32>("get_state", (), 1).await.ok()?;
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();
        let my_buffer: &mut [u8] = &mut [0; 8];
        memory
            .read(&mut self.store, json_string_ptr as usize, my_buffer)
            .unwrap();

        let json_string_ptr: *const PointerLengthString =
            my_buffer.as_ptr() as *const PointerLengthString;
        let json_string: &PointerLengthString = unsafe { &*json_string_ptr };

        let json_string = get_string_from_memory(
            &memory,
            &mut self.store,
            json_string.ptr as i32,
            json_string.len as i32,
        );
        if json_string.is_none() {
            println!("No json string");
        }
        json_string
    }


}
