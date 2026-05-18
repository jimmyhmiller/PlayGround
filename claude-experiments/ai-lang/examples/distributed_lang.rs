//! Two-process demo where the *ai-lang program itself* calls `at`.
//!
//! Compared to `distributed.rs` (where Rust harness code shipped the
//! closure), this binary's Rust glue is dumber: it compiles the
//! source, installs the runtime as the thread's current Runtime, and
//! calls a single ai-lang def. *That def's body* — written in
//! ai-lang — invokes `at(node, || work(x))`, which is a builtin
//! lowered to a call into the network layer.
//!
//! Usage:
//!
//! ```text
//! # Terminal A (server):
//! cargo run --example distributed_lang -- server 19999
//!
//! # Terminal B (client):
//! cargo run --example distributed_lang -- client 192 168 0 55 19999 42
//! ```
//!
//! The client args are `<a> <b> <c> <d> <port> <x>`. The first five
//! are passed to `make_node` in the program; `x` is the input to the
//! work function. Server reachability is your problem: same source
//! must be compiled on both sides (no NeedCode/Code yet), and the
//! server's TCP port must be reachable from the client.

use std::env;
use std::sync::Once;

use ai_lang::codegen::{CompiledModule, Jit, def_symbol, init_native_target};
use ai_lang::hash::Hash;
use ai_lang::knowledge::KnowledgeBase;
use ai_lang::net::{
    NetError, RuntimeHandle, accept_one, bind, build_at_runtime_binding,
    clear_current_at_binding, clear_current_knowledge_base, clear_current_runtime,
    install_current_at_binding, install_current_knowledge_base, install_current_runtime,
    serve_one,
};
use ai_lang::parser::parse_module;
use ai_lang::resolve::{ResolvedModule, resolve_module};
use ai_lang::runtime::{Runtime, Thread};
use inkwell::context::Context;

/// The shared program both server and client compile. The program
/// itself contains the `at(...)` call — the Rust harness only kicks
/// off `do_remote_work` (or its equivalent server-side entry).
///
/// `Node { a, b, c, d, port }` represents an IPv4 socket. The runtime
/// `ai_net_at` reads the five Int fields by offset.
const PROGRAM: &str = "
    struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
    enum Failure {
        Unreachable(Node),
        Crashed(Node),
        CodeMissing(Node),
        Cancelled(Node),
    }
    enum Result<T, E> { Ok(T), Err(E) }

    struct Pair { x: Int, y: Int }
    enum Mode { Double(Int), Triple(Int) }

    def square(n: Int) -> Int = n * n

    def transform(m: Mode) -> Int = match m {
        Double(x) => x * 2,
        Triple(x) => x * 3,
    }

    def combine(p: Pair) -> Int =
        transform(Double(p.x)) + transform(Triple(p.y)) - 7

    def work(a: Int, b: Int) -> Int = {
        let p = Pair { x: square(a), y: square(b) };
        combine(p)
    }

    def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
        Node { a: a, b: b, c: c, d: d, port: port }

    // The headline def: the ai-lang program ITSELF calls `at`, which
    // returns Result<Int, Failure>. We match to recover the Int (or
    // return -1 on any failure).
    def do_remote_work(node: Node, a: Int, b: Int) -> Int =
        match at(node, || work(a, b)) {
            Ok(n) => n,
            Err(_) => 0 - 1,
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
) -> (
    Runtime,
    Jit<'ctx>,
    std::collections::HashMap<String, Hash>,
    ResolvedModule,
) {
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
    (rt, jit, names, r)
}

fn run_server(port: u16) -> Result<(), NetError> {
    init_llvm();
    let addr = format!("0.0.0.0:{}", port);
    let ctx = Context::create();
    let (rt, _jit, _names, _resolved) = build_runtime(&ctx);
    install_current_runtime(&rt);

    let listener = bind(&addr)?;
    let local = listener.local_addr()?;
    eprintln!("[server] listening on {}", local);

    let _handle = std::sync::Arc::new(RuntimeHandle(rt));
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

fn run_client(a: i64, b: i64, c: i64, d: i64, port: i64, x: i64) -> Result<(), NetError> {
    init_llvm();
    let ctx = Context::create();
    let (rt, jit, names, resolved) = build_runtime(&ctx);
    install_current_runtime(&rt);
    // The legacy demo: both sides share the same program, so the server
    // never asks for code. Install an empty KB just to satisfy
    // `ai_net_at`'s precondition.
    let kb = KnowledgeBase::new();
    install_current_knowledge_base(&kb);
    let resolver_binding = resolved.at_binding.as_ref().expect("at_binding populated");
    let rt_binding = build_at_runtime_binding(&rt, resolver_binding)
        .expect("runtime at-binding built");
    install_current_at_binding(&rt_binding);

    // We don't have a single-arg-and-Node-builder API at the Rust
    // level; instead, build a *Node value* via the ai-lang `make_node`
    // def, then invoke `do_remote_work(node, x, x)`. This keeps the
    // Rust harness from doing anything beyond marshalling Ints in and
    // an Int back.
    let make_node_sym = def_symbol(&names["make_node"]);
    let make_node = unsafe {
        jit.engine
            .get_function::<unsafe extern "C" fn(
                *mut Thread,
                i64,
                i64,
                i64,
                i64,
                i64,
            ) -> *mut u8>(&make_node_sym)
            .expect("make_node JIT'd")
    };
    let node_ptr = unsafe { make_node.call(rt.thread_ptr(), a, b, c, d, port) };
    eprintln!(
        "[client] built Node {{ a:{} b:{} c:{} d:{} port:{} }}; calling do_remote_work(node, {x}, {x})",
        a, b, c, d, port,
    );

    let do_remote_sym = def_symbol(&names["do_remote_work"]);
    let do_remote = unsafe {
        jit.engine
            .get_function::<unsafe extern "C" fn(
                *mut Thread,
                *mut u8,
                i64,
                i64,
            ) -> i64>(&do_remote_sym)
            .expect("do_remote_work JIT'd")
    };

    // The ai-lang code inside `do_remote_work` calls `at(node, ...)`.
    // That goes through ai_net_at, which TCP-connects to the Node's
    // address, ships the closure, and reads back the Int.
    let result = unsafe { do_remote.call(rt.thread_ptr(), node_ptr, x, x) };

    // Print just the number (ai-lang code shipped + ran + returned).
    println!("{}", result);
    let expected = 2 * x * x + 3 * x * x - 7;
    eprintln!(
        "[client] received result: {} (expected: {})",
        result, expected
    );

    clear_current_runtime();
    clear_current_knowledge_base();
    clear_current_at_binding();
    Ok(())
}

fn print_usage() {
    eprintln!("usage:");
    eprintln!("  distributed_lang server <port>");
    eprintln!("  distributed_lang client <a> <b> <c> <d> <port> <x>");
    eprintln!();
    eprintln!("  client builds a Node value for IPv4 a.b.c.d:port inside");
    eprintln!("  ai-lang, then ai-lang's `do_remote_work` calls");
    eprintln!("  `at(node, || work(x, x))`. Server runs the closure and");
    eprintln!("  returns the Int. Formula: 2x² + 3x² - 7 = 5x² - 7.");
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
            if args.len() < 8 {
                print_usage();
                std::process::exit(2);
            }
            let parse = |i: usize| -> i64 {
                args[i].parse().unwrap_or_else(|_| {
                    eprintln!("error: arg {} must be an integer", i);
                    std::process::exit(2);
                })
            };
            run_client(parse(2), parse(3), parse(4), parse(5), parse(6), parse(7))
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
