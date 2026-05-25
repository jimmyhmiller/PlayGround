//! Smoke test for the WASM component toolchain.
//!
//! Loads the prebuilt `counter.component.wasm` from `tests/wasm/`, wires
//! its two imports (`get-count` and `set-count`) to a `RefCell<u32>`, and
//! calls its `handle(event: u32)` export a few times. Verifies the
//! counter rises as expected.
//!
//! To rebuild the component:
//!
//!     cd wasm-samples/counter
//!     cargo build --target wasm32-unknown-unknown --release
//!     wasm-tools component new \
//!         target/wasm32-unknown-unknown/release/counter.wasm \
//!         -o counter.component.wasm
//!     cp counter.component.wasm \
//!         ../../crates/runtime/tests/wasm/counter.component.wasm

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use wasmtime::component::{Component, Linker};
use wasmtime::{Engine, Store};

fn component_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/wasm/counter.component.wasm")
}

struct HostState {
    _marker: (),
}

#[test]
fn counter_increments_via_wasm_component() {
    let engine = Engine::default();
    let component = Component::from_file(&engine, component_path()).expect("load component");

    let counter = Arc::new(Mutex::new(0u32));
    let mut linker = Linker::<HostState>::new(&engine);
    {
        let mut root = linker.root();
        let counter_for_get = counter.clone();
        root.func_wrap("get-count", move |_store, ()| -> wasmtime::Result<(u32,)> {
            let v = *counter_for_get.lock().unwrap();
            Ok((v,))
        })
        .expect("wrap get-count");

        let counter_for_set = counter.clone();
        root.func_wrap(
            "set-count",
            move |_store, (value,): (u32,)| -> wasmtime::Result<()> {
                *counter_for_set.lock().unwrap() = value;
                Ok(())
            },
        )
        .expect("wrap set-count");
    }

    let mut store = Store::new(&engine, HostState { _marker: () });
    let instance = linker
        .instantiate(&mut store, &component)
        .expect("instantiate");

    // Look up `handle` from the instance's exports.
    let handle = instance
        .get_typed_func::<(u32,), ()>(&mut store, "handle")
        .expect("get handle export");

    handle.call(&mut store, (5,)).expect("call 1");
    handle.call(&mut store, (3,)).expect("call 2");
    handle.call(&mut store, (10,)).expect("call 3");

    let final_value = *counter.lock().unwrap();
    assert_eq!(
        final_value, 18,
        "counter should have accumulated 5 + 3 + 10"
    );
}
