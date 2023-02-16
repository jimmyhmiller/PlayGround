use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::{thread, time::Duration, sync::Arc};

use futures::{executor, task::SpawnExt};
use futures::FutureExt;
use futures_timer::Delay;
use wasmtime::{Instance, Store, Engine, Config, Linker, Module};

struct State {
    wasi: wasmtime_wasi::WasiCtx,

}

impl State {
    fn new(wasi: wasmtime_wasi::WasiCtx) -> Self {
        Self { wasi }
    }
}


struct WasmInstance {
    instance: Instance,
    store: Store<State>,
}

impl WasmInstance {
    async fn new(engine: Arc<Engine>) -> Result<Self, Box<dyn std::error::Error>>{

        let wasm_path = "/Users/jimmyhmiller/Documents/Code/PlayGround/rust/wasm-infinite-loop/target/wasm32-wasi/debug/wasm_infinite_loop.wasm";


        let wasi = wasmtime_wasi::WasiCtxBuilder::new()
            .inherit_stdio()
            .inherit_stderr()
            .inherit_env()?
            .build();

        let mut linker: Linker<State> = Linker::new(&engine);
        wasmtime_wasi::add_to_linker(&mut linker, |s| &mut s.wasi)?;

        let mut store = Store::new(&engine, State::new(wasi));
        let module = Module::from_file(&engine, wasm_path)?;

        let instance = linker.instantiate_async(&mut store, &module).await?;
        Ok(Self { instance, store })
    }

    async fn init(&mut self, label: i32) -> Result<(), Box<dyn std::error::Error>> {
        self.store.epoch_deadline_async_yield_and_update(1);
        let init = self.instance.get_typed_func::<i32, i32>(&mut self.store, "loop_forever")?;
        let result = init.call_async(&mut self.store, label).await?;
        println!("Result: {}", result);
        Ok(())
    }

}


fn main() -> Result<(), Box<dyn std::error::Error>> {



    let mut config = Config::new();
    config.async_support(true);
    config.epoch_interruption(true);

    let engine = Arc::new(Engine::new(&config).unwrap());

    let engine_clone = engine.clone();
    thread::spawn(move || {
        loop {
            engine_clone.increment_epoch();
            thread::sleep(Duration::from_nanos(1));
        }
    });

    async fn run(label: i32, engine: Arc<Engine>) -> Result<(), Box<dyn std::error::Error>> {
        let mut instance = WasmInstance::new(engine).await.unwrap();
        instance.init(label).await.unwrap();
        Ok(())
    }


    let mut local_pool = futures::executor::LocalPool::new();
    let local_spawner = local_pool.spawner();

    local_spawner.spawn(run(1, engine.clone()).map(|_| ())).unwrap();
    local_spawner.spawn(run(2, engine.clone()).map(|_| ())).unwrap();

    println!("==================");
    for _ in 0..100 {
        local_pool.run_until(Delay::new(Duration::from_nanos(1)));
        println!("Ran");
    }

    println!("done");


    Ok(())
}
