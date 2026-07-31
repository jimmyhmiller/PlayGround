//! A **real Node native addon**, instrumented with memscope — the fixture for
//! the "Rust built as a library, used from a Node app" shape.
//!
//! This is the shape that breaks every assumption a profiler makes about
//! programs: there is no `main`, the executable is `node` (which has no Rust
//! debug info at all), and the code we care about lives in a `.node` file dyld
//! loaded at runtime. memscope handles it because a recording symbolicates
//! against **the image memscope is compiled into**, found via `dladdr`, not
//! against `current_exe()`.
//!
//! The N-API surface is hand-written against the C API rather than via napi-rs:
//! a test fixture shouldn't pull a binding framework (and its whole crates.io
//! tree) in just to export one function. What node sees is identical.
//!
//! Build + drive it:
//! ```sh
//! cargo build -p nodeaddon --release
//! cp target/release/libnodeaddon.dylib target/release/nodeaddon.node
//! MEMSCOPE_RECORD=/tmp/addon.mscope node crates/nodeaddon/driver.js target/release/nodeaddon.node 200
//! memscope analyze /tmp/addon.mscope
//! ```

use std::ffi::c_void;
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::sync::Mutex;

/// The tracking allocator. In a cdylib this covers the Rust code compiled into
/// *this module* — which is exactly the code we want to profile; node's own
/// allocations are none of our business.
#[global_allocator]
static GLOBAL: memscope::MemScope = memscope::MemScope::system();

// --- the N-API bits we need --------------------------------------------------

type NapiEnv = *mut c_void;
type NapiValue = *mut c_void;
type NapiCallbackInfo = *mut c_void;
type NapiCallback = extern "C" fn(NapiEnv, NapiCallbackInfo) -> NapiValue;

/// `NAPI_AUTO_LENGTH`: "the name is NUL-terminated, measure it yourself".
const NAPI_AUTO_LENGTH: usize = usize::MAX;

extern "C" {
    fn napi_create_function(
        env: NapiEnv,
        utf8name: *const c_char,
        length: usize,
        cb: NapiCallback,
        data: *mut c_void,
        result: *mut NapiValue,
    ) -> c_int;
    fn napi_set_named_property(
        env: NapiEnv,
        object: NapiValue,
        utf8name: *const c_char,
        value: NapiValue,
    ) -> c_int;
    fn napi_get_undefined(env: NapiEnv, result: *mut NapiValue) -> c_int;
}

// --- the workload ------------------------------------------------------------

/// A type with a name worth recovering: the test asserts DWARF gives us
/// `nodeaddon::Widget` out of a `.node` file loaded by `node`.
struct Widget {
    id: u64,
    name: String,
    samples: Vec<u32>,
}

/// Widgets the addon never frees — an addon-side leak, which is the thing you
/// reach for a profiler to find. These are what the recording's live set holds
/// at the end.
static KEPT: Mutex<Vec<Box<Widget>>> = Mutex::new(Vec::new());

fn make_widget(id: u64) -> Widget {
    Widget {
        id,
        name: format!("widget-{id}"),
        samples: (0..32).map(|i| i as u32 * 3).collect(),
    }
}

/// One call's worth of work: churn a batch of widgets, keep one forever.
fn do_work(call: u64) {
    // Every allocation below is tagged with these, so `memscope flamegraph
    // --group-by call` works on an addon exactly as on a program.
    let _m = memscope::meta!(subsystem = "widgets", call = call);

    let batch: Vec<Widget> = (0..16).map(|i| make_widget(call * 16 + i)).collect();
    let total: u64 =
        batch.iter().map(|w| w.id + w.samples.len() as u64 + w.name.len() as u64).sum();
    // Keep the work from being optimized away, and keep one widget alive.
    if total > 0 {
        KEPT.lock().unwrap().push(Box::new(make_widget(call)));
    }
    drop(batch);
}

// --- the addon ---------------------------------------------------------------

extern "C" fn work(env: NapiEnv, _info: NapiCallbackInfo) -> NapiValue {
    static CALLS: Mutex<u64> = Mutex::new(0);
    let call = {
        let mut c = CALLS.lock().unwrap();
        *c += 1;
        *c
    };
    do_work(call);

    let mut undefined: NapiValue = ptr::null_mut();
    // SAFETY: `env` is the live napi_env node handed us on this JS thread.
    unsafe { napi_get_undefined(env, &mut undefined) };
    undefined
}

/// Node's module entry point: dyld loads the `.node`, node calls this.
///
/// The memscope integration is the **one** `memscope::init()` call. It reads
/// `MEMSCOPE_*` from the environment and does nothing at all when none are set,
/// so an addon can ship with this line permanently in place.
///
/// # Safety
/// Called by node with a valid `env` and the module's `exports` object.
#[no_mangle]
pub unsafe extern "C" fn napi_register_module_v1(env: NapiEnv, exports: NapiValue) -> NapiValue {
    memscope::init();

    let mut f: NapiValue = ptr::null_mut();
    let name = b"work\0".as_ptr() as *const c_char;
    if napi_create_function(env, name, NAPI_AUTO_LENGTH, work, ptr::null_mut(), &mut f) == 0 {
        napi_set_named_property(env, exports, name, f);
    }
    exports
}
