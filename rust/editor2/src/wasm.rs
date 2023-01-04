use std::error::Error;
use skia_safe::Canvas;
use wasmtime::{self, Engine, Linker, Module, Store, Caller, AsContextMut, Memory};
use wasmtime_wasi::{WasiCtxBuilder, WasiCtx};


struct PointerLengthString {
    ptr: u32,
    len: u32,
}


unsafe fn _any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

enum Commands {
    DrawRect(f32, f32, f32, f32),
    DrawString(String, f32, f32),
}

struct State {
    wasi: WasiCtx,
    commands: Vec<Commands>,
}

impl State {
    fn new(wasi: WasiCtx) -> Self {
        Self {
            wasi,
            commands: Vec::new(),
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
    let data = mem.into_memory().unwrap().data(store)
        .get(ptr as u32 as usize..)
        .and_then(|arr| arr.get(..len as u32 as usize));
    let string = from_utf8(data.unwrap()).unwrap();
    string.to_string()
}

fn get_string_from_memory(memory: &Memory, store: &mut Store<State>, ptr: i32, len: i32) -> String {
    use core::str::from_utf8;
    let data = memory.data(store)
        .get(ptr as u32 as usize..)
        .and_then(|arr| arr.get(..len as u32 as usize));
    let string = from_utf8(data.unwrap()).unwrap();
    string.to_string()
}



impl WasmContext {
    pub fn new(wasm_path: &str) -> Result<Self, Box<dyn Error>> {
        // An engine stores and configures global compilation settings like
        // optimization level, enabled wasm features, etc.
        let engine = Engine::default();

        let mut linker : Linker<State> = Linker::new(&engine);
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
        linker.func_wrap("host", "draw_rect", |mut caller: Caller<'_, State>, x: f32, y: f32, width: f32, height: f32| {
            let state = caller.data_mut();
            state.commands.push(Commands::DrawRect(x, y, width, height));
        })?;
        linker.func_wrap("host", "draw_str", |mut caller: Caller<'_, State>, ptr: i32, len: i32, x: f32, y: f32| {
            let string = get_string_from_caller(&mut caller, ptr, len);
            let state = caller.data_mut();
            state.commands.push(Commands::DrawString(string, x, y));
        })?;
        Ok(())
    }

    pub fn draw(&mut self, canvas: &mut Canvas) -> Result<(), Box<dyn Error>> {

        if let Some(func) = self.instance.get_func(&mut self.store, "draw") {

            let func = func.typed::<(), ()>(&mut self.store)?;
            func.call(&mut self.store, ())?;
            let state = &mut self.store.data_mut();
            for command in state.commands.iter() {
                match command {
                    Commands::DrawRect(x, y, width, height) => {
                        canvas.draw_rect(skia_safe::Rect::from_xywh(*x, *y, *width, *height), &skia_safe::Paint::default());
                    }
                    Commands::DrawString(str, x, y) => {
                        let mut paint = skia_safe::Paint::default();
                        paint.set_color(skia_safe::Color::WHITE);
                        let mut font = skia_safe::Font::default();
                        font.set_size(30.0);
                        canvas.draw_str(str, (*x, *y), &font, &paint);
                    }
                }
            }
            state.commands.clear();
        } else {
            println!("No draw function");
        }


        Ok(())
    }

    pub fn on_click(&mut self) -> Result<(), Box<dyn Error>> {

        if let Some(func) = self.instance.get_func(&mut self.store, "on_click") {
            let func = func.typed::<(), ()>(&mut self.store)?;
            func.call(&mut self.store, ())?;
        } else {
            println!("No on_click function");
        }
        Ok(())
    }

    pub fn reload(&mut self) -> Result<(), Box<dyn Error>> {
        if let Some(func) = self.instance.get_func(&mut self.store, "get_state") {
            println!("{:?}", func.ty(&self.store));
            let func = func.typed::<(), i32>(&mut self.store)?;
            let json_string_ptr = func.call(&mut self.store, ())?;
            let memory = self.instance.get_export(&mut self.store, "memory").unwrap().into_memory().unwrap();
            let my_buffer: &mut [u8] = &mut [0; 8];
            memory.read(&mut self.store, json_string_ptr as usize, my_buffer).unwrap();

            let json_string_ptr: *const PointerLengthString = my_buffer.as_ptr() as *const PointerLengthString;
            let json_string: &PointerLengthString = unsafe { &*json_string_ptr };

            let json_string = get_string_from_memory(&memory, &mut self.store, json_string.ptr as i32, json_string.len as i32);
            let data = json_string.as_bytes();

                // Do I need to set memory?
            let module = Module::from_file(&self.engine, &self.path)?;
            let instance = self.linker.instantiate(&mut self.store, &module)?;
            let memory = instance.get_export(&mut self.store, "memory").unwrap().into_memory().unwrap();
            self.instance = instance;
            memory.write(&mut self.store, 0, &data).unwrap();
            if let Some(func) = self.instance.get_func(&mut self.store, "set_state") {
                println!("{:?}", func.ty(&self.store));
                let func = func.typed::<(i32, i32), ()>(&mut self.store)?;
                func.call(&mut self.store, (0, data.len() as i32))?;
            }
        }



        Ok(())
    }
}

