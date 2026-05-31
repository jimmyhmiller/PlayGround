//! Distributed Game of Life — every cell is its own OS process,
//! every server starts with **zero code**, code fetches over the wire
//! via the `NeedCode` round-trip, and TCP connections are persistent.
//!
//! Run:
//!   cargo run --release --example game_of_life_distributed
//!
//! That spawns this same binary 49 times with `--server`, one per cell.
//! Verify with `ps`:
//!   ps -ef | grep game_of_life_distributed
//! You'll see 50 distinct PIDs (1 client + 49 servers).
//!
//! Controls while running (type + enter):
//!   <number>   set frame delay in ms (e.g., 30 = fast, 1000 = slow)
//!   q          quit (kills all child servers)
//!
//! ## What the servers know on startup
//!
//! Nothing. Each child process boots, builds an *empty* `Runtime` +
//! `IncrementalJit` (parse "" → empty ResolvedModule → empty
//! CompiledModule), binds a port, prints `READY <port>` on stdout for
//! the parent to read, then serves forever.
//!
//! When the client ships its first closure to a server, `decode_value`
//! fails because the closure's shape isn't registered. The server
//! replies with `NeedCode([closure_hash])`; the client looks the hash
//! up in its `KnowledgeBase` (which it built up-front from the resolved
//! module), collects transitive deps, ships them as a `Code` frame.
//! The server's `IncrementalJit::install` lifts each Def + Lambda into
//! its live LLVM JIT, then retries decode → succeeds → invokes →
//! replies with `Result(Int)`. Subsequent calls reuse the now-installed
//! code AND the same TCP connection.

use ai_lang::Hash;
use ai_lang::ast::Type;
use ai_lang::codegen::{
    CompiledModule, IncrementalJit, Jit, def_symbol, init_native_target,
};
use ai_lang::ffi::register_extern;
use ai_lang::knowledge::KnowledgeBase;
use ai_lang::net::{
    bind, build_at_runtime_binding, clear_at_conn_cache, clear_current_at_binding,
    clear_current_knowledge_base, clear_current_runtime, install_current_at_binding,
    install_current_knowledge_base, install_current_runtime, serve_with_install,
};
use ai_lang::parser::parse_module;
use ai_lang::resolve::resolve_module;
use ai_lang::runtime::{Runtime, Thread};
use ai_lang::stdlib::SOURCE as STDLIB;
use ai_lang::typecheck::{TypeCache, typecheck_module};
use inkwell::context::Context;
use std::collections::HashMap;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

const WIDTH: i64 = 7;
const HEIGHT: i64 = 7;
const NUM_CELLS: usize = (WIDTH * HEIGHT) as usize;

static PORTS: OnceLock<Vec<u16>> = OnceLock::new();

#[unsafe(no_mangle)]
unsafe extern "C" fn ai_demo_get_port(_thread: *mut Thread, i: i64) -> i64 {
    PORTS
        .get()
        .and_then(|v| v.get(i as usize).copied())
        .map(|p| p as i64)
        .unwrap_or(-1)
}

const USER_SRC: &str = r#"
    // Node / Failure / Result / tcp_node come from the stdlib.

    extern fn get_port(i: Int) -> Int

    def build_pool(n: Int) -> List<Node> =
        list_reverse(build_pool_acc(n, 0, Nil))

    def build_pool_acc(n: Int, i: Int, acc: List<Node>) -> List<Node> =
        if i >= n { acc } else {
            let p = get_port(i);
            build_pool_acc(
                n, i + 1,
                Cons(ListCell { head: tcp_node(127, 0, 0, 1, p), tail: acc })
            )
        }

    def wrap(a: Int, m: Int) -> Int = {
        let r = a - (a / m) * m;
        if r < 0 { r + m } else { r }
    }

    def bit_at(packed: Int, i: Int) -> Int = {
        let p = pow(2, i);
        let shifted = packed / p;
        shifted - (shifted / 2) * 2
    }

    def set_bit_one(packed: Int, i: Int) -> Int = {
        let cur = bit_at(packed, i);
        if cur == 1 { packed } else { packed + pow(2, i) }
    }

    def cell_at(packed: Int, x: Int, y: Int) -> Int =
        bit_at(packed, wrap(y, 7) * 7 + wrap(x, 7))

    def count_neighbours(packed: Int, x: Int, y: Int) -> Int =
        cell_at(packed, x - 1, y - 1) + cell_at(packed, x, y - 1) +
        cell_at(packed, x + 1, y - 1) +
        cell_at(packed, x - 1, y) + cell_at(packed, x + 1, y) +
        cell_at(packed, x - 1, y + 1) + cell_at(packed, x, y + 1) +
        cell_at(packed, x + 1, y + 1)

    def compute_cell(packed: Int, x: Int, y: Int) -> Int = {
        let alive = cell_at(packed, x, y);
        let n = count_neighbours(packed, x, y);
        if alive == 1 {
            if n == 2 { 1 } else { if n == 3 { 1 } else { 0 } }
        } else {
            if n == 3 { 1 } else { 0 }
        }
    }

    def remote_cell(pool: List<Node>, packed: Int, x: Int, y: Int) -> Int = {
        let i = y * 7 + x;
        match list_at(pool, i) {
            Some(node) => match at(node, || compute_cell(packed, x, y)) {
                Ok(v) => v,
                Err(_) => 0,
            },
            None => 0,
        }
    }

    def distributed_step(pool: List<Node>, packed: Int) -> Int =
        distributed_step_acc(pool, packed, 0, 0)

    def distributed_step_acc(pool: List<Node>, src: Int, i: Int, acc: Int) -> Int =
        if i >= 49 { acc } else {
            let x = i - (i / 7) * 7;
            let y = i / 7;
            let bit = remote_cell(pool, src, x, y);
            let next = if bit == 1 { set_bit_one(acc, i) } else { acc };
            distributed_step_acc(pool, src, i + 1, next)
        }

    def step_one_gen(pool: List<Node>, packed: Int) -> Int =
        distributed_step(pool, packed)

    def initial_glider() -> Int =
        pow(2, 1) + pow(2, 9) + pow(2, 14) + pow(2, 15) + pow(2, 16)
"#;

fn full_src() -> String {
    format!("{}\n{}", STDLIB, USER_SRC)
}

// =============================================================================
// Server mode — empty Runtime, install code on demand.
// =============================================================================

fn run_server() {
    init_native_target().expect("init native target");

    // Empty runtime. Parse "", resolve, build CompiledModule. The
    // resulting CompiledModule has no defs, no lambdas, no closure
    // shapes — it's just the LLVM module with runtime-extern decls
    // (`ai_gc_alloc_closure`, `ai_gc_box_int`, etc.) wired through.
    let ctx = Context::create();
    let empty_m = parse_module("").expect("parse empty");
    let empty_r = resolve_module(&empty_m).expect("resolve empty");
    let empty_cm = CompiledModule::build(&ctx, &empty_r).expect("build empty");
    let mut rt = Runtime::new_with_metadata(
        empty_cm.closure_type_infos.clone(),
        empty_cm.shape_registry.clone(),
        empty_cm.shape_meta.clone(),
        empty_cm.shape_by_type_id.clone(),
    );
    let mut jit = IncrementalJit::new(empty_cm, &rt).expect("incremental jit");

    let listener = bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().unwrap().port();

    // Signal port to parent (single line on stdout).
    println!("READY {}", port);
    let _ = io::stdout().flush();

    // Watchdog thread: if parent's stdin pipe to us closes, the parent
    // died — exit cleanly so we don't become an orphan.
    thread::spawn(|| {
        let stdin = io::stdin();
        let mut buf = [0u8; 64];
        loop {
            match stdin.lock().read(&mut buf) {
                Ok(0) | Err(_) => {
                    std::process::exit(0);
                }
                Ok(_) => {} // ignore any actual data
            }
        }
    });

    eprintln!("[server pid={} port={}] empty runtime, awaiting code", std::process::id(), port);

    // Serve forever, INLINE (IncrementalJit is !Send so we can't move
    // it to a worker thread). Each accepted stream stays open and is
    // reused for many Call/Result cycles via the inner loop.
    loop {
        let mut stream = match listener.accept() {
            Ok((s, _)) => s,
            Err(_) => break,
        };
        // Disable Nagle so per-request latency is low.
        let _ = stream.set_nodelay(true);
        loop {
            match unsafe { serve_with_install(&mut rt, &mut jit, &mut stream) } {
                Ok(()) => continue,
                Err(_) => break,
            }
        }
    }
}

// =============================================================================
// Client mode — spawn 49 children, build KB, run sim.
// =============================================================================

/// RAII wrapper that kills the spawned server on drop. Belt-and-
/// suspenders for the stdin-watchdog: if the parent exits via panic
/// or signal, the OS closes the child's stdin → child exits. If the
/// parent exits cleanly, this Drop kills them explicitly.
struct ServerChild {
    child: Child,
    port: u16,
}

impl Drop for ServerChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_server_process(idx: usize) -> io::Result<ServerChild> {
    let exe = std::env::current_exe()?;
    let mut child = Command::new(exe)
        .arg("--server")
        .arg(format!("--idx={}", idx))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    // Read the READY line from child's stdout.
    let stdout = child.stdout.take().expect("child stdout piped");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let port: u16 = line
        .trim()
        .strip_prefix("READY ")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("server #{} bad READY line: {:?}", idx, line),
            )
        })?;
    // Drop the reader; we don't need to read further. Child's stdin
    // pipe stays held in `child.stdin` (we never write to it; on parent
    // exit it closes and the child's watchdog kicks in).
    drop(reader);
    Ok(ServerChild { child, port })
}

fn render(packed: i64, generation: u64, delay_ms: u64, server_pids: &[u32]) -> String {
    let mut out = String::with_capacity(512);
    out.push_str("\x1b[2J\x1b[H");
    out.push_str("  Distributed Game of Life (toroidal 7x7)\n");
    out.push_str(&format!(
        "  Generation {}   |   delay {}ms\n",
        generation, delay_ms,
    ));
    out.push_str(&format!(
        "  49 OS server processes, each started with ZERO code (code fetched via NeedCode).\n",
    ));
    let head_pids: Vec<String> = server_pids.iter().take(6).map(|p| p.to_string()).collect();
    out.push_str(&format!(
        "  Server PIDs (first 6 of {}): {} ...\n",
        server_pids.len(),
        head_pids.join(", "),
    ));
    out.push_str("\n");
    out.push_str("  +");
    for _ in 0..WIDTH {
        out.push_str("--");
    }
    out.push_str("-+\n");
    for y in 0..HEIGHT {
        out.push_str("  | ");
        for x in 0..WIDTH {
            let i = (y * WIDTH + x) as u32;
            let bit = (packed >> i) & 1;
            out.push_str(if bit == 1 { "\x1b[32m█\x1b[0m " } else { "· " });
        }
        out.push_str("|\n");
    }
    out.push_str("  +");
    for _ in 0..WIDTH {
        out.push_str("--");
    }
    out.push_str("-+\n");
    out.push_str(&format!("  live: {}\n", packed.count_ones()));
    out.push_str("\n");
    out.push_str("  Type <ms><enter> to change delay, q<enter> to quit.\n");
    out
}

fn run_client() {
    init_native_target().expect("init native target");

    // Host extern for the client (servers don't need this — the closures
    // shipped to them don't transitively depend on `build_pool`).
    unsafe {
        register_extern(
            "get_port",
            vec![Type::Builtin("Int".to_owned())],
            Type::Builtin("Int".to_owned()),
            ai_demo_get_port as usize,
        );
    }

    let src = full_src();

    // Typecheck up-front.
    {
        let m = parse_module(&src).expect("parse");
        let r = resolve_module(&m).expect("resolve");
        let mut cache = TypeCache::new();
        typecheck_module(&r, &mut cache).expect("typecheck");
    }

    // ---- Spawn 49 server CHILDREN PROCESSES. ----
    eprintln!("Spawning {} server processes...", NUM_CELLS);
    let t0 = std::time::Instant::now();
    let mut servers: Vec<ServerChild> = Vec::with_capacity(NUM_CELLS);
    let mut ports: Vec<u16> = Vec::with_capacity(NUM_CELLS);
    for i in 0..NUM_CELLS {
        let sc = spawn_server_process(i).expect("spawn server");
        ports.push(sc.port);
        servers.push(sc);
    }
    let server_pids: Vec<u32> = servers.iter().map(|s| s.child.id()).collect();
    eprintln!(
        "  {} server processes ready in {:.1}s. PIDs {:?}",
        NUM_CELLS,
        t0.elapsed().as_secs_f64(),
        &server_pids[..6.min(server_pids.len())],
    );
    PORTS.set(ports.clone()).expect("PORTS init");

    // ---- Client runtime + JIT. ----
    let client_ctx = Context::create();
    let m = parse_module(&src).expect("parse client");
    let r = resolve_module(&m).expect("resolve client");
    let names: HashMap<String, Hash> =
        r.defs.iter().map(|d| (d.name.clone(), d.hash)).collect();
    let cm = CompiledModule::build(&client_ctx, &r).expect("build client");
    let client_rt = Runtime::new_with_metadata(
        cm.closure_type_infos.clone(),
        cm.shape_registry.clone(),
        cm.shape_meta.clone(),
        cm.shape_by_type_id.clone(),
    );
    let client_jit = Jit::new(cm, &client_rt).expect("jit");

    // ---- Knowledge base: every def + every reachable lambda. ----
    // This is what gets shipped to servers on NeedCode.
    let kb = KnowledgeBase::build(&r);
    eprintln!("KnowledgeBase: {} items", kb.len());

    install_current_runtime(&client_rt);
    install_current_knowledge_base(&kb);
    let resolver_binding = r.at_binding.as_ref().expect("at_binding populated");
    let rt_binding =
        build_at_runtime_binding(&client_rt, resolver_binding).expect("rt binding");
    install_current_at_binding(&rt_binding);

    let build_pool = unsafe {
        client_jit
            .engine
            .get_function::<unsafe extern "C" fn(*mut Thread, i64) -> *mut u8>(
                &def_symbol(&names["build_pool"]),
            )
            .expect("build_pool symbol")
    };
    let initial = unsafe {
        client_jit
            .engine
            .get_function::<unsafe extern "C" fn(*mut Thread) -> i64>(
                &def_symbol(&names["initial_glider"]),
            )
            .expect("initial_glider symbol")
    };
    let step = unsafe {
        client_jit
            .engine
            .get_function::<unsafe extern "C" fn(*mut Thread, *mut u8, i64) -> i64>(
                &def_symbol(&names["step_one_gen"]),
            )
            .expect("step_one_gen symbol")
    };

    let pool = unsafe { build_pool.call(client_rt.thread_ptr(), NUM_CELLS as i64) };

    // ---- Control state. ----
    let delay_ms = Arc::new(AtomicU64::new(250));
    let running = Arc::new(AtomicBool::new(true));

    {
        let delay_ms = delay_ms.clone();
        let running = running.clone();
        thread::Builder::new()
            .name("stdin-reader".to_owned())
            .spawn(move || {
                let stdin = io::stdin();
                let mut line = String::new();
                while running.load(Ordering::Relaxed) {
                    line.clear();
                    match stdin.lock().read_line(&mut line) {
                        Ok(0) => break,
                        Ok(_) => {}
                        Err(_) => break,
                    }
                    let trimmed = line.trim();
                    if trimmed.eq_ignore_ascii_case("q")
                        || trimmed.eq_ignore_ascii_case("quit")
                        || trimmed.eq_ignore_ascii_case("exit")
                    {
                        running.store(false, Ordering::Relaxed);
                        break;
                    }
                    if let Ok(ms) = trimmed.parse::<u64>() {
                        delay_ms.store(ms.max(10).min(60_000), Ordering::Relaxed);
                    }
                }
            })
            .expect("spawn stdin reader");
    }

    // ---- Forever loop. ----
    let mut grid = unsafe { initial.call(client_rt.thread_ptr()) };
    let mut gen_counter: u64 = 0;
    print!(
        "{}",
        render(grid, gen_counter, delay_ms.load(Ordering::Relaxed), &server_pids)
    );
    let _ = io::stdout().flush();

    while running.load(Ordering::Relaxed) {
        let ms = delay_ms.load(Ordering::Relaxed);
        thread::sleep(Duration::from_millis(ms));
        if !running.load(Ordering::Relaxed) {
            break;
        }
        grid = unsafe { step.call(client_rt.thread_ptr(), pool, grid) };
        gen_counter += 1;
        print!(
            "{}",
            render(grid, gen_counter, delay_ms.load(Ordering::Relaxed), &server_pids)
        );
        let _ = io::stdout().flush();
    }

    println!(
        "\n  Bye. {} generations × 49 cells = {} remote at() calls across {} processes.",
        gen_counter,
        gen_counter * 49,
        NUM_CELLS,
    );

    clear_at_conn_cache();
    clear_current_runtime();
    clear_current_knowledge_base();
    clear_current_at_binding();

    // `servers` drops here → each ServerChild::drop kills its child.
    drop(servers);
}

// =============================================================================
// main: dispatch on --server.
// =============================================================================

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let is_server = args.iter().any(|a| a == "--server");
    if is_server {
        run_server();
    } else {
        run_client();
    }
}
