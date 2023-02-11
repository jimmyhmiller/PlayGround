use skia_safe::{Canvas, Font, FontStyle, Typeface};
use std::{error::Error, collections::HashMap, hash::Hash, mem};
use wasmtime::{self, AsContextMut, Caller, Engine, Linker, Memory, Module, Store};
use wasmtime_wasi::{WasiCtx, WasiCtxBuilder};

use crate::widget::{Size, Color};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
struct PointerLengthString {
    ptr: u32,
    len: u32,
}

unsafe fn _any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts((p as *const T) as *const u8, ::std::mem::size_of::<T>())
}

#[derive(Debug)]
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

struct State {
    wasi: WasiCtx,
    commands: Vec<Command>,
    // Probably not the best structure
    // but lets start here
    process_messages: HashMap<i32, String>,
}

impl State {
    fn new(wasi: WasiCtx) -> Self {
        Self {
            wasi,
            commands: Vec::new(),
            process_messages: HashMap::new(),
        }
    }
}

pub struct WasmContext {
    instance: wasmtime::Instance,
    store: Store<State>,
    engine: Engine,
    path: String,
    linker: Linker<State>,
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

fn get_string_from_memory(memory: &Memory, store: &mut Store<State>, ptr: i32, len: i32) -> Option<String> {
    use core::str::from_utf8;
    let data = memory
        .data(store)
        .get(ptr as u32 as usize..)
        .and_then(|arr| arr.get(..len as u32 as usize));
    let string = from_utf8(data.unwrap()).ok()?;
    Some(string.to_string())
}

impl WasmContext {
    pub fn new(wasm_path: &str) -> Result<Self, Box<dyn Error>> {
        // An engine stores and configures global compilation settings like
        // optimization level, enabled wasm features, etc.
        let engine = Engine::default();

        let mut linker: Linker<State> = Linker::new(&engine);
        wasmtime_wasi::add_to_linker(&mut linker, |s| &mut s.wasi)?;

        let wasi = WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_args()?
            .build();

        Self::setup_host_functions(&mut linker)?;

        let mut store = Store::new(&engine, State::new(wasi));
        let module = Module::from_file(&engine, wasm_path)?;

        let instance = linker.instantiate(&mut store, &module)?;

        Ok(Self {
            path: wasm_path.to_string(),
            instance,
            linker,
            store,
            engine,
        })
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
            |mut caller: Caller<'_, State>, x: f32, y: f32, width: f32, height: f32, radius: f32| {
                let state = caller.data_mut();
                state.commands.push(Command::DrawRRect(x, y, width, height, radius));
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
        linker.func_wrap(
            "host",
            "save",
            |mut caller: Caller<'_, State>| {
                let state = caller.data_mut();
                state.commands.push(Command::Save);
            },
        )?;
        linker.func_wrap(
            "host",
            "restore",
            |mut caller: Caller<'_, State>| {
                let state = caller.data_mut();
                state.commands.push(Command::Restore);
            },
        )?;
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
            |mut caller: Caller<'_, State>, ptr: i32, len: i32| -> i32{
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
                state.commands.push(Command::SendProcessMessage(process_id, message));
            },
        )?;

        linker.func_wrap(
            "host",
            "recieve_last_message_low_level",
            |mut caller: Caller<'_, State>, ptr: i32, process_id: i32| {
                {
                    let state = caller.data_mut();
                    state.commands.push(Command::ReceiveLastProcessMessage(process_id));
                }
                let state = caller.data_mut();
                let message = state.process_messages.get(&process_id).unwrap_or(&"test".to_string()).clone();
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

    pub fn draw(&mut self, fn_name: &str, canvas: &mut Canvas) -> Result<Size, Box<dyn Error>> {
        let mut max_width = 0.0;
        let mut max_height = 0.0;
        if let Some(func) = self.instance.get_func(&mut self.store, fn_name) {
            let func = func.typed::<(), ()>(&mut self.store)?;
            func.call(&mut self.store, ())?;
            let state = &mut self.store.data_mut();

            let mut paint = skia_safe::Paint::default();

            for command in state.commands.iter() {
                match command {
                    Command::SetColor(r, g, b, a) => {
                        paint.set_color(Color::new(*r, *g, *b, *a).to_color4f().to_color());
                    }
                    Command::DrawRect(x, y, width, height) => {
                        canvas.draw_rect(
                            skia_safe::Rect::from_xywh(*x, *y, *width, *height),
                            &paint,
                        );
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
            state.commands.clear();
        } else {
            println!("No {} function", fn_name
        );
        }

        Ok(Size {
            width: max_width,
            height: max_height,
        })
    }

    pub fn on_click(&mut self, x: f32, y: f32) -> Result<(), Box<dyn Error>> {
        if let Some(func) = self.instance.get_func(&mut self.store, "on_click") {
            let func = func.typed::<(f32, f32), ()>(&mut self.store)?;
            func.call(&mut self.store, (x, y))?;
        } else {
            println!("No on_click function");
        }
        Ok(())
    }

    pub fn on_scroll(&mut self, x: f64, y: f64) -> Result<(), Box<dyn Error>> {
        if let Some(func) = self.instance.get_func(&mut self.store, "on_scroll") {
            let func = func.typed::<(f64, f64), ()>(&mut self.store)?;
            func.call(&mut self.store, (x, y))?;
        } else {
            println!("No on_scroll function");
        }
        Ok(())
    }

    pub fn on_key(&mut self, key_code: u32, state: u32, modifiers: u32) -> Result<(), Box<dyn Error>> {
        if let Some(func) = self.instance.get_func(&mut self.store, "on_key") {
            let func = func.typed::<(u32, u32, u32), ()>(&mut self.store)?;
            func.call(&mut self.store, (key_code, state, modifiers))?;
        } else {
            println!("No on_key function");
        }
        Ok(())
    }

    pub fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        let json_string = self.get_state().ok_or("no get state function")?;
        let data = json_string.as_bytes();

        let module = Module::from_file(&self.engine, &self.path)?;
        let instance = self.linker.instantiate(&mut self.store, &module)?;
        self.instance = instance;
        self.set_state(data)?;

        Ok(())
    }

    pub fn set_state(&mut self, data: &[u8]) -> Result<(), Box<dyn Error>> {
        let memory = self
            .instance
            .get_export(&mut self.store, "memory")
            .unwrap()
            .into_memory()
            .unwrap();
        memory.write(&mut self.store, 0, &data).unwrap();
        let func = self
            .instance
            .get_func(&mut self.store, "set_state")
            .ok_or("no function set_state")?;
        let func = func.typed::<(i32, i32), ()>(&mut self.store)?;
        func.call(&mut self.store, (0, data.len() as i32))?;
        Ok(())
    }

    pub fn get_state(&mut self) -> Option<String> {
        let func = self.instance.get_func(&mut self.store, "get_state")?;
        let func = func.typed::<(), i32>(&mut self.store).ok()?;
        let json_string_ptr = func.call(&mut self.store, ()).ok()?;
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
