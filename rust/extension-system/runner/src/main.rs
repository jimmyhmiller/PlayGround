use std::error::Error;
use wasmtime::{self, Engine, Linker, Module, Store, WasmParams, Caller, AsContextMut};
use wasmtime_wasi::{WasiCtxBuilder, WasiCtx};


#[repr(C)]
pub struct MyStruct {
    pub a: i32,
    pub b: i32,
}


unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    ::std::slice::from_raw_parts(
        (p as *const T) as *const u8,
        ::std::mem::size_of::<T>(),
    )
}

struct State {
    count: i32,
    wasi: WasiCtx,
}

impl State {
    fn new(wasi: WasiCtx) -> Self {
        Self {
            count: 0,
            wasi,
        }
    }

    fn increment(&mut self) {
        self.count += 1;
    }
}



fn main() -> Result<(), Box<dyn Error>> {
    // An engine stores and configures global compilation settings like
    // optimization level, enabled wasm features, etc.
    let engine = Engine::default();

    let mut linker : Linker<State> = Linker::new(&engine);
    wasmtime_wasi::add_to_linker(&mut linker, |s| &mut s.wasi)?;


    linker.func_wrap("host", "foo", |mut caller: Caller<'_, State>, offset: i32| {

        let memory = caller.get_export("memory").unwrap().into_memory().unwrap();
        let buffer = &mut [0u8; 8];
        memory.read(caller.as_context_mut(), offset as usize, buffer).unwrap();
        let s = unsafe { std::mem::transmute::<[u8; 8], MyStruct>(*buffer) };
        println!("a {} b {}", s.a, s.b);
        caller.data_mut().increment();

        s.a + s.b
    })?;



    // We start off by creating a `Module` which represents a compiled form
    // of our input wasm module. In this case it'll be JIT-compiled after
    // we parse the text format.


    // A `Store` is what will own instances, functions, globals, etc. All wasm
    // items are stored within a `Store`, and it's what we'll always be using to
    // interact with the wasm world. Custom data can be stored in stores but for
    // now we just use `()`.
    let wasi = WasiCtxBuilder::new()
        .inherit_stdio()
        .inherit_args()?
        .inherit_env()?

        .build();

    let state = State::new(wasi);

    let mut store = Store::new(&engine, state);

    loop {
        let module = Module::from_file(&engine, "./target/wasm32-wasi/debug/my_first_extension.wasm")?;

        let instance = linker.instantiate(&mut store, &module)?;

        // The `Instance` gives us access to various exported functions and items,
        // which we access here to pull out our `answer` exported function and
        // run it.
        let answer = instance.get_func(&mut store, "answer")
            .expect("`answer` was not an exported function");
        println!("here #{:?}", answer);

        // There's a few ways we can call the `answer` `Func` value. The easiest
        // is to statically assert its signature with `typed` (in this case
        // asserting it takes no arguments and returns one i32) and then call it.
        let answer = answer.typed::<i32, i32>(&store)?;
        println!("here2");

        let my_struct = MyStruct{ a: 42, b: 42};

        let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or("error")?;


        memory.write(&mut store, 0, unsafe { any_as_u8_slice(&my_struct) }).unwrap();

        println!("state: {:?}", store.data().count);

        // And finally we can call our function! Note that the error propagation
        // with `?` is done to handle the case where the wasm function traps.
        let result = answer.call(&mut store,0)?;
        println!("Answer: {:?}", result);

        println!("state: {:?}", store.data().count);
        // wait for input, if empty exit
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().is_empty() {
            break;
        }

    }
    Ok(())
}
