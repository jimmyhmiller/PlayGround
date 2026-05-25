//! Two-process distributed demo.
//!
//! Usage:
//!
//! ```text
//! # Terminal A:
//! cargo run --example distributed -- server 9999
//!
//! # Terminal B:
//! cargo run --example distributed -- client 127.0.0.1:9999 42
//! ```
//!
//! Both processes compile the same source. The client builds a
//! closure capturing `n=42`, ships it over TCP to the server. The
//! server invokes the closure (`|| n + 100`) and returns the result.
//! The client prints `142`.
//!
//! ## What this proves
//!
//! - **Closures are shippable.** A value carrying a code reference
//!   (32-byte content hash) + captured environment, encoded as bytes,
//!   reconstructed on a remote heap.
//! - **Code is identified by content.** The server looks up the
//!   lambda's JIT'd entry point by hash, not by name or symbol. Both
//!   sides compiled the same canonical AST, so both arrived at the
//!   same hash for the same code.
//! - **GC stays correct across the wire.** The server allocates the
//!   incoming closure in its own heap; its own shadow stack roots
//!   keep it alive through any collection during execution.

use std::env;
use std::sync::Arc;
use std::sync::Once;

use ai_lang::codegen::{CompiledModule, Jit, def_symbol, init_native_target};
use ai_lang::hash::Hash;
use ai_lang::knowledge::KnowledgeBase;
use ai_lang::net::{
    NetError, RuntimeHandle, accept_one, at_remote, bind, serve_one,
};
use ai_lang::parser::parse_module;
use ai_lang::resolve::resolve_module;
use ai_lang::runtime::{Runtime, Thread};
use inkwell::context::Context;

/// The shared program both server and client compile. Identical source
/// → identical canonical AST → identical content hashes → both sides
/// agree on which JIT'd function a closure's code_hash refers to.
///
/// The closure built by `make_work` captures two Ints and, when
/// invoked on the server, runs a body that exercises every value type
/// we've shipped:
///
/// - `let` bindings (sequential SSA on the lifted body)
/// - struct construction + field access (heap-allocated `Pair`)
/// - enum construction + `match` with payload binding (`Mode::Double` /
///   `Mode::Triple`)
/// - cross-def calls (`square`, `transform`, `combine`) — each a TopRef
///   resolved by hash, so the server hits its own JIT'd entry points
/// - arithmetic over Int values
///
/// Formula computed: `2 * a² + 3 * b² - 7`.
///   (3, 4) →  59
///   (5, 5) → 118
///   (10, 1) → 196
const PROGRAM: &str = "
    struct Pair { x: Int, y: Int }
    enum Mode { Double(Int), Triple(Int) }

    def square(n: Int) -> Int = n * n

    def transform(m: Mode) -> Int = match m {
        Double(x) => x * 2,
        Triple(x) => x * 3,
    }

    def combine(p: Pair) -> Int =
        transform(Double(p.x)) + transform(Triple(p.y)) - 7

    def make_work(a: Int, b: Int) -> fn() -> Int = || {
        let p = Pair { x: square(a), y: square(b) };
        combine(p)
    }
";

static INIT: Once = Once::new();
fn init_llvm() {
    INIT.call_once(|| {
        init_native_target().expect("init native target");
    });
}

fn build_runtime<'ctx>(
    ctx: &'ctx Context,
) -> (Runtime, Jit<'ctx>, std::collections::HashMap<String, Hash>) {
    let m = parse_module(PROGRAM).unwrap();
    let r = resolve_module(&m).unwrap();
    let names: std::collections::HashMap<String, Hash> =
        r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
    let cm = CompiledModule::build(ctx, &r).unwrap();
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let jit = Jit::new(cm, &rt).unwrap();
    (rt, jit, names)
}

fn run_server(port: u16) -> Result<(), NetError> {
    init_llvm();
    let addr = format!("0.0.0.0:{}", port);
    let ctx = Context::create();
    let (rt, _jit, _names) = build_runtime(&ctx);

    let listener = bind(&addr)?;
    let local = listener.local_addr()?;
    eprintln!("[server] listening on {}", local);

    // Wrap the runtime so the connection handler doesn't need
    // unsafe-Send shenanigans; here we serve in-line on the main
    // thread of this process, one connection at a time.
    let _handle = Arc::new(RuntimeHandle(rt));
    let rt_ref: &Runtime = &_handle.0;

    loop {
        eprintln!("[server] waiting for connection...");
        let mut stream = accept_one(&listener)?;
        eprintln!("[server] accepted; serving one Call");
        match unsafe { serve_one(rt_ref, &mut stream) } {
            Ok(()) => eprintln!("[server] reply sent"),
            Err(e) => eprintln!("[server] error: {}", e),
        }
    }
}

fn run_client(addr: &str, a: i64, b: i64) -> Result<(), NetError> {
    init_llvm();
    let ctx = Context::create();
    let (rt, jit, names) = build_runtime(&ctx);

    let make_work_hash = names["make_work"];
    let make_work = unsafe {
        jit.engine
            .get_function::<unsafe extern "C" fn(*mut Thread, i64, i64) -> *mut u8>(
                &def_symbol(&make_work_hash),
            )
            .expect("make_work JIT'd")
    };
    let closure = unsafe { make_work.call(rt.thread_ptr(), a, b) };
    eprintln!(
        "[client] built closure capturing a={}, b={}; shipping to {}",
        a, b, addr
    );
    eprintln!(
        "[client] code_hash for make_work's lambda body: {}",
        unsafe { closure_code_hash(closure as *const u8) }
    );

    // The other side of `distributed.rs` is the legacy demo where both
    // sides have compiled the same source, so the server doesn't need
    // any code from us. An empty KB is sufficient.
    let kb = KnowledgeBase::new();
    let result = unsafe { at_remote(&rt, &kb, addr, closure as *const u8)? };
    println!("{}", result);
    let expected = 2 * a * a + 3 * b * b - 7;
    eprintln!("[client] received result: {} (expected: {})", result, expected);
    Ok(())
}

/// Read the 32-byte code_hash stored at offset `Full::SIZE` from the
/// start of a closure heap object.
unsafe fn closure_code_hash(ptr: *const u8) -> Hash {
    use ai_lang::gc::{Full, ObjHeader};
    let mut h = [0u8; 32];
    unsafe {
        core::ptr::copy_nonoverlapping(
            ptr.add(<Full as ObjHeader>::SIZE),
            h.as_mut_ptr(),
            32,
        );
    }
    Hash(h)
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  distributed server <port>");
    eprintln!("  distributed client <host:port> <a> <b>");
    eprintln!();
    eprintln!("  client builds a closure capturing (a, b), ships it to the");
    eprintln!("  server, prints the result of running it remotely.");
    eprintln!("  result formula: 2*a² + 3*b² - 7");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(2);
    }
    let result = match args[1].as_str() {
        "server" => {
            let port: u16 = args
                .get(2)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    print_usage();
                    std::process::exit(2);
                });
            run_server(port)
        }
        "client" => {
            if args.len() < 5 {
                print_usage();
                std::process::exit(2);
            }
            let addr = &args[2];
            let a: i64 = args[3].parse().unwrap_or_else(|_| {
                eprintln!("error: <a> must be an integer");
                std::process::exit(2);
            });
            let b: i64 = args[4].parse().unwrap_or_else(|_| {
                eprintln!("error: <b> must be an integer");
                std::process::exit(2);
            });
            run_client(addr, a, b)
        }
        _ => {
            print_usage();
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {}", e);
        std::process::exit(1);
    }
}
