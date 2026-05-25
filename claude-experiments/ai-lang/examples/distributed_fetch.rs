//! Full code-fetch demo: server starts with NO code, client ships a
//! closure, server requests the code via `NeedCode`, client responds
//! with the canonical bytes (def + lambda + transitive struct deps),
//! server installs them into a live JIT and invokes the closure.
//!
//! ## Usage
//!
//! ```text
//! # Terminal A (server):
//! cargo run --example distributed_fetch -- server 20000
//!
//! # Terminal B (client):
//! cargo run --example distributed_fetch -- client 127 0 0 1 20000 42
//! ```
//!
//! Client args are `<a> <b> <c> <d> <port> <x>`. The first five build
//! the IPv4:port pair the client connects to; `x` is the input to the
//! work function. The server prints what it's installing as it learns
//! more code; the client prints the final result.

use std::env;
use std::sync::Once;

use ai_lang::codegen::{CompiledModule, IncrementalJit, Jit, def_symbol, init_native_target};
use ai_lang::hash::Hash;
use ai_lang::knowledge::KnowledgeBase;
use ai_lang::net::{
    NetError, accept_one, bind, build_at_runtime_binding, clear_current_at_binding,
    clear_current_knowledge_base, clear_current_runtime, install_current_at_binding,
    install_current_knowledge_base, install_current_runtime, serve_with_install,
};
use ai_lang::parser::parse_module;
use ai_lang::resolve::resolve_module;
use ai_lang::runtime::{Runtime, Thread};
use inkwell::context::Context;

/// The CLIENT's program. The server starts with an empty program.
///
/// `do_remote_work` runs locally (on the client) and contains the
/// `at(node, || work(x))` call. The closure `|| work(x)` is what gets
/// shipped — the server doesn't see `do_remote_work` and doesn't need
/// to. It only needs to install:
///   - `work` (because the lambda body references it via TopRef)
///   - the lambda itself (its closure shape + JIT'd body)
///
/// `make_node`/`Node` are also installed because the closure's wire
/// encoding carries them: actually NO — the closure's captures are
/// just an Int (`x`), so only `work` and the lambda need to ship.
const CLIENT_PROGRAM: &str = "
    struct Node { a: Int, b: Int, c: Int, d: Int, port: Int }
    enum Failure {
        Unreachable(Node),
        Crashed(Node),
        CodeMissing(Node),
        Cancelled(Node),
    }
    enum Result<T, E> { Ok(T), Err(E) }

    def work(x: Int) -> Int = x * x + 7

    def make_node(a: Int, b: Int, c: Int, d: Int, port: Int) -> Node =
        Node { a: a, b: b, c: c, d: d, port: port }

    def do_remote_work(node: Node, x: Int) -> Int =
        match at(node, || work(x)) {
            Ok(n) => n,
            Err(_) => 0 - 1,
        }
";

/// The SERVER's program: completely empty. The server's runtime is
/// constructed with zero defs, zero closures, zero shapes. It learns
/// everything it needs from the wire.
const SERVER_PROGRAM: &str = "";

static INIT: Once = Once::new();
fn init_llvm() {
    INIT.call_once(|| {
        init_native_target().expect("init native target");
    });
}

fn run_server(port: u16) -> Result<(), NetError> {
    init_llvm();
    let addr = format!("0.0.0.0:{}", port);

    let ctx = Context::create();
    let m = parse_module(SERVER_PROGRAM).expect("parse empty program");
    let r = resolve_module(&m).expect("resolve empty program");
    let cm = CompiledModule::build(&ctx, &r).expect("build empty module");
    let mut rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let mut jit = IncrementalJit::new(cm, &rt).expect("init IncrementalJit");

    let listener = bind(&addr)?;
    let local = listener.local_addr()?;
    eprintln!(
        "[server] listening on {} — runtime starts with 0 defs, 0 shapes",
        local
    );

    loop {
        eprintln!("[server] waiting for connection...");
        let mut stream = accept_one(&listener)?;
        eprintln!(
            "[server] accepted; current code_table has {} entries; type_table has {} shapes",
            // CodeTable has no public len; just print shape count.
            "?",
            rt.shape_by_type_id.len()
        );
        match unsafe { serve_with_install(&mut rt, &mut jit, &mut stream) } {
            Ok(()) => eprintln!(
                "[server] reply sent. Now have {} shapes registered.",
                rt.shape_by_type_id.len()
            ),
            Err(e) => eprintln!("[server] error: {}", e),
        }
    }
}

fn run_client(a: i64, b: i64, c: i64, d: i64, port: i64, x: i64) -> Result<(), NetError> {
    init_llvm();
    let ctx = Context::create();
    let m = parse_module(CLIENT_PROGRAM).expect("parse client program");
    let r = resolve_module(&m).expect("resolve client program");
    let names: std::collections::HashMap<String, Hash> =
        r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
    let cm = CompiledModule::build(&ctx, &r).expect("build client module");
    let rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let _jit = Jit::new(cm, &rt).expect("init JIT");

    // Build a knowledge base from the full resolved module — every def
    // + every reachable lambda — so we can answer NeedCode requests.
    let kb = KnowledgeBase::build(&r);
    eprintln!(
        "[client] knowledge base built with {} items (defs + lambdas)",
        kb.len()
    );

    install_current_runtime(&rt);
    install_current_knowledge_base(&kb);
    let resolver_binding = r.at_binding.as_ref().expect("at_binding populated");
    let rt_binding = build_at_runtime_binding(&rt, resolver_binding)
        .expect("runtime at-binding built");
    install_current_at_binding(&rt_binding);

    // Build the Node via the lang def.
    let make_node = unsafe {
        _jit.engine
            .get_function::<unsafe extern "C" fn(
                *mut Thread,
                i64,
                i64,
                i64,
                i64,
                i64,
            ) -> *mut u8>(&def_symbol(&names["make_node"]))
            .expect("make_node JIT'd")
    };
    let node_ptr = unsafe { make_node.call(rt.thread_ptr(), a, b, c, d, port) };
    eprintln!(
        "[client] Node = {{ a:{} b:{} c:{} d:{} port:{} }}; calling do_remote_work(node, {})",
        a, b, c, d, port, x
    );

    // Invoke do_remote_work — this is what triggers the at(node, || work(x))
    // call internally, which uses the runtime fn `ai_net_at`, which uses
    // the installed Runtime + KB to ship the closure.
    let do_remote = unsafe {
        _jit.engine
            .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                &def_symbol(&names["do_remote_work"]),
            )
            .expect("do_remote_work JIT'd")
    };
    let result = unsafe { do_remote.call(rt.thread_ptr(), node_ptr, x) };

    println!("{}", result);
    let expected = x * x + 7;
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
    eprintln!("  distributed_fetch server <port>");
    eprintln!("  distributed_fetch client <a> <b> <c> <d> <port> <x>");
    eprintln!();
    eprintln!("  Server starts with an EMPTY runtime (no defs, no shapes).");
    eprintln!("  Client ships `|| work(x)` to the server. The server requests");
    eprintln!("  the code, installs it incrementally, then runs the closure.");
    eprintln!("  Formula: work(x) = x*x + 7.");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(2);
    }
    let result = match args[1].as_str() {
        "server" => {
            let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or_else(|| {
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
            run_client(
                parse(2),
                parse(3),
                parse(4),
                parse(5),
                parse(6),
                parse(7),
            )
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
